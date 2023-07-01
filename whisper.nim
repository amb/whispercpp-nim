
const
  WHISPER_SAMPLE_RATE* = 16000
  WHISPER_N_FFT* = 400
  WHISPER_N_MEL* = 80
  WHISPER_HOP_LENGTH* = 160
  WHISPER_CHUNK_SIZE* = 30

{.link: "whisper.lib".}

##
##  C interface
##
##  The following interface is thread-safe as long as the sample whisper_context is not used by multiple threads
##  concurrently.
##
##  Basic usage:
##
##      #include "whisper.h"
##
##      ...
##
##      struct whisper_context * ctx = whisper_init_from_file("/path/to/ggml-base.en.bin");
##
##      if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
##          fprintf(stderr, "failed to process audio\n");
##          return 7;
##      }
##
##      const int n_segments = whisper_full_n_segments(ctx);
##      for (int i = 0; i < n_segments; ++i) {
##          const char * text = whisper_full_get_segment_text(ctx, i);
##          printf("%s", text);
##      }
##
##      whisper_free(ctx);
##
##      ...
##
##  This is a demonstration of the most straightforward usage of the library.
##  "pcmf32" contains the RAW audio data in 32-bit floating point format.
##
##  The interface also allows for more fine-grained control over the computation, but it requires a deeper
##  understanding of how the model works.
##

type
  whisper_context* = cint
  whisper_state* = cint
  whisper_token* = cint
  whisper_token_data* {.bycopy.} = object
    id*: whisper_token
    ##  token id
    tid*: whisper_token
    ##  forced timestamp token id
    p*: cfloat
    ##  probability of the token
    plog*: cfloat
    ##  log probability of the token
    pt*: cfloat
    ##  probability of the timestamp token
    ptsum*: cfloat
    ##  sum of probabilities of all timestamp tokens
    ##  token-level timestamp data
    ##  do not use if you haven't computed token-level timestamps
    t0*: int64
    ##  start time of the token
    t1*: int64
    ##    end time of the token
    vlen*: cfloat
    ##  voice length of the token

  whisper_model_loader* {.bycopy.} = object
    context*: pointer
    read*: proc (ctx: pointer; output: pointer; read_size: csize_t): csize_t {.cdecl.}
    eof*: proc (ctx: pointer): bool {.cdecl.}
    close*: proc (ctx: pointer) {.cdecl.}

##  Available sampling strategies
type
  whisper_sampling_strategy* {.size: sizeof(cint).} = enum
    WHISPER_SAMPLING_GREEDY,     ##  similar to OpenAI's GreedyDecoder
    WHISPER_SAMPLING_BEAM_SEARCH ##  similar to OpenAI's BeamSearchDecoder


##  Text segment callback
##  Called on every newly generated text segment
##  Use the whisper_full_...() functions to obtain the text segments
type
  whisper_new_segment_callback* = proc (ctx: ptr whisper_context;
                                     state: ptr whisper_state; n_new: cint;
                                     user_data: pointer) {.cdecl.}

##  Progress callback
type
  whisper_progress_callback* = proc (ctx: ptr whisper_context;
                                  state: ptr whisper_state; progress: cint;
                                  user_data: pointer) {.cdecl.}

##  Encoder begin callback
##  If not NULL, called before the encoder starts
##  If it returns false, the computation is aborted
type
  whisper_encoder_begin_callback* = proc (ctx: ptr whisper_context;
                                       state: ptr whisper_state; user_data: pointer): bool {.cdecl.}

##  Logits filter callback
##  Can be used to modify the logits before sampling
##  If not NULL, called after applying temperature to logits
type
  whisper_logits_filter_callback* = proc (ctx: ptr whisper_context;
                                       state: ptr whisper_state;
                                       tokens: ptr whisper_token_data;
                                       n_tokens: cint; logits: ptr cfloat;
                                       user_data: pointer) {.cdecl.}

##  Parameters for the whisper_full() function
##  If you chnage the order or add new parameters, make sure to update the default values in whisper.cpp:
##  whisper_full_default_params()
type
  INNER_C_STRUCT_whisper_0* {.bycopy.} = object
    best_of*: cint
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264

  INNER_C_STRUCT_whisper_1* {.bycopy.} = object
    beam_size*: cint
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265
    patience*: cfloat
    ##  TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf

  whisper_full_params* {.bycopy.} = object
    strategy*: whisper_sampling_strategy
    n_threads*: cint
    n_max_text_ctx*: cint
    ##  max tokens to use from past text as prompt for the decoder
    offset_ms*: cint
    ##  start offset in ms
    duration_ms*: cint
    ##  audio duration to process in ms
    translate*: bool
    no_context*: bool
    ##  do not use past transcription (if any) as initial prompt for the decoder
    single_segment*: bool
    ##  force single segment output (useful for streaming)
    print_special*: bool
    ##  print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
    print_progress*: bool
    ##  print progress information
    print_realtime*: bool
    ##  print results from within whisper.cpp (avoid it, use callback instead)
    print_timestamps*: bool
    ##  print timestamps for each text segment when printing realtime
    ##  [EXPERIMENTAL] token-level timestamps
    token_timestamps*: bool
    ##  enable token-level timestamps
    thold_pt*: cfloat
    ##  timestamp token probability threshold (~0.01)
    thold_ptsum*: cfloat
    ##  timestamp token sum probability threshold (~0.01)
    max_len*: cint
    ##  max segment length in characters
    split_on_word*: bool
    ##  split on word rather than on token (when used with max_len)
    max_tokens*: cint
    ##  max tokens per segment (0 = no limit)
    ##  [EXPERIMENTAL] speed-up techniques
    ##  note: these can significantly reduce the quality of the output
    speed_up*: bool
    ##  speed-up the audio by 2x using Phase Vocoder
    audio_ctx*: cint
    ##  overwrite the audio context size (0 = use default)
    ##  tokens to provide to the whisper decoder as initial prompt
    ##  these are prepended to any existing text context from a previous call
    initial_prompt*: cstring
    prompt_tokens*: ptr whisper_token
    prompt_n_tokens*: cint
    ##  for auto-detection, set to nullptr, "" or "auto"
    language*: cstring
    detect_language*: bool
    ##  common decoding parameters:
    suppress_blank*: bool
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
    suppress_non_speech_tokens*: bool
    ##  ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
    temperature*: cfloat
    ##  initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
    max_initial_ts*: cfloat
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
    length_penalty*: cfloat
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267
    ##  fallback parameters
    ##  ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
    temperature_inc*: cfloat
    entropy_thold*: cfloat
    ##  similar to OpenAI's "compression_ratio_threshold"
    logprob_thold*: cfloat
    no_speech_thold*: cfloat
    ##  TODO: not implemented
    greedy*: INNER_C_STRUCT_whisper_0
    beam_search*: INNER_C_STRUCT_whisper_1
    ##  called for every newly generated text segment
    new_segment_callback*: whisper_new_segment_callback
    new_segment_callback_user_data*: pointer
    ##  called on each progress update
    progress_callback*: whisper_progress_callback
    progress_callback_user_data*: pointer
    ##  called each time before the encoder starts
    encoder_begin_callback*: whisper_encoder_begin_callback
    encoder_begin_callback_user_data*: pointer
    ##  called by each decoder to filter obtained logits
    logits_filter_callback*: whisper_logits_filter_callback
    logits_filter_callback_user_data*: pointer


##  Various functions for loading a ggml whisper model.
##  Allocate (almost) all memory needed for the model.
##  Return NULL on failure
proc whisper_init_from_file*(path_model: cstring): ptr whisper_context {.cdecl, importc.}
proc whisper_init_from_buffer*(buffer: pointer; buffer_size: csize_t): ptr whisper_context {.cdecl, importc.}
proc whisper_init*(loader: ptr whisper_model_loader): ptr whisper_context {.cdecl, importc.}

##  These are the same as the above, but the internal state of the context is not allocated automatically
##  It is the responsibility of the caller to allocate the state using whisper_init_state() (#523)
proc whisper_init_from_file_no_state*(path_model: cstring): ptr whisper_context {.cdecl, importc.}
proc whisper_init_from_buffer_no_state*(buffer: pointer;
    buffer_size: csize_t): ptr whisper_context {.cdecl, importc.}
proc whisper_init_no_state*(loader: ptr whisper_model_loader): ptr whisper_context {.cdecl, importc.}
proc whisper_init_state*(ctx: ptr whisper_context): ptr whisper_state {.cdecl, importc.}

##  Frees all allocated memory
proc whisper_free*(ctx: ptr whisper_context) {.cdecl, importc.}
proc whisper_free_state*(state: ptr whisper_state) {.cdecl, importc.}
proc whisper_free_params*(params: ptr whisper_full_params) {.cdecl, importc.}

##  Convert RAW PCM audio to log mel spectrogram.
##  The resulting spectrogram is stored inside the default state of the provided whisper context.
##  Returns 0 on success
proc whisper_pcm_to_mel*(ctx: ptr whisper_context; samples: ptr cfloat;
                        n_samples: cint; n_threads: cint): cint {.cdecl, importc.}
proc whisper_pcm_to_mel_with_state*(ctx: ptr whisper_context;
                                   state: ptr whisper_state; samples: ptr cfloat;
                                   n_samples: cint; n_threads: cint): cint {.cdecl, importc.}

##  Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
##  The resulting spectrogram is stored inside the default state of the provided whisper context.
##  Returns 0 on success
proc whisper_pcm_to_mel_phase_vocoder*(ctx: ptr whisper_context;
                                      samples: ptr cfloat; n_samples: cint;
                                      n_threads: cint): cint {.cdecl, importc.}
proc whisper_pcm_to_mel_phase_vocoder_with_state*(ctx: ptr whisper_context;
    state: ptr whisper_state; samples: ptr cfloat; n_samples: cint; n_threads: cint): cint {.cdecl, importc.}

##  This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
##  Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
##  n_mel must be 80
##  Returns 0 on success
proc whisper_set_mel*(ctx: ptr whisper_context; data: ptr cfloat; n_len: cint;
                     n_mel: cint): cint {.cdecl, importc.}
proc whisper_set_mel_with_state*(ctx: ptr whisper_context; state: ptr whisper_state;
                                data: ptr cfloat; n_len: cint; n_mel: cint): cint {.cdecl, importc.}

##  Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
##  Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
##  offset can be used to specify the offset of the first frame in the spectrogram.
##  Returns 0 on success
proc whisper_encode*(ctx: ptr whisper_context; offset: cint; n_threads: cint): cint {.cdecl, importc.}
proc whisper_encode_with_state*(ctx: ptr whisper_context; state: ptr whisper_state;
                               offset: cint; n_threads: cint): cint {.cdecl, importc.}

##  Run the Whisper decoder to obtain the logits and probabilities for the next token.
##  Make sure to call whisper_encode() first.
##  tokens + n_tokens is the provided context for the decoder.
##  n_past is the number of tokens to use from previous decoder calls.
##  Returns 0 on success
##  TODO: add support for multiple decoders
proc whisper_decode*(ctx: ptr whisper_context; tokens: ptr whisper_token;
                    n_tokens: cint; n_past: cint; n_threads: cint): cint {.cdecl, importc.}
proc whisper_decode_with_state*(ctx: ptr whisper_context; state: ptr whisper_state;
                               tokens: ptr whisper_token; n_tokens: cint;
                               n_past: cint; n_threads: cint): cint {.cdecl, importc.}

##  Convert the provided text into tokens.
##  The tokens pointer must be large enough to hold the resulting tokens.
##  Returns the number of tokens on success, no more than n_max_tokens
##  Returns -1 on failure
##  TODO: not sure if correct
proc whisper_tokenize*(ctx: ptr whisper_context; text: cstring;
                      tokens: ptr whisper_token; n_max_tokens: cint): cint {.cdecl, importc.}

##  Largest language id (i.e. number of available languages - 1)
proc whisper_lang_max_id*(): cint {.cdecl, importc.}

##  Return the id of the specified language, returns -1 if not found
##  Examples:
##    "de" -> 2
##    "german" -> 2
proc whisper_lang_id*(lang: cstring): cint {.cdecl, importc.}

##  Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
proc whisper_lang_str*(id: cint): cstring {.cdecl, importc.}

##  Use mel data at offset_ms to try and auto-detect the spoken language
##  Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
##  Returns the top language id or negative on failure
##  If not null, fills the lang_probs array with the probabilities of all languages
##  The array must be whisper_lang_max_id() + 1 in size
##  ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
proc whisper_lang_auto_detect*(ctx: ptr whisper_context; offset_ms: cint;
                              n_threads: cint; lang_probs: ptr cfloat): cint {.cdecl, importc.}
proc whisper_lang_auto_detect_with_state*(ctx: ptr whisper_context;
    state: ptr whisper_state; offset_ms: cint; n_threads: cint; lang_probs: ptr cfloat): cint {.cdecl, importc.}
proc whisper_n_len*(ctx: ptr whisper_context): cint {.cdecl, importc.}

##  mel length
proc whisper_n_len_from_state*(state: ptr whisper_state): cint {.cdecl, importc.}

##  mel length
proc whisper_n_vocab*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_n_text_ctx*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_n_audio_ctx*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_is_multilingual*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_vocab*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_audio_ctx*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_audio_state*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_audio_head*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_audio_layer*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_text_ctx*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_text_state*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_text_head*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_text_layer*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_n_mels*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_ftype*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_model_type*(ctx: ptr whisper_context): cint {.cdecl, importc.}

##  Token logits obtained from the last call to whisper_decode()
##  The logits for the last token are stored in the last row
##  Rows: n_tokens
##  Cols: n_vocab
proc whisper_get_logits*(ctx: ptr whisper_context): ptr cfloat {.cdecl, importc.}
proc whisper_get_logits_from_state*(state: ptr whisper_state): ptr cfloat {.cdecl, importc.}

##  Token Id -> String. Uses the vocabulary in the provided context
proc whisper_token_to_str*(ctx: ptr whisper_context; token: whisper_token): cstring {.cdecl, importc.}
proc whisper_model_type_readable*(ctx: ptr whisper_context): cstring {.cdecl, importc.}

##  Special tokens
proc whisper_token_eot*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_sot*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_prev*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_solm*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_not*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_beg*(ctx: ptr whisper_context): whisper_token {.cdecl, importc.}
proc whisper_token_lang*(ctx: ptr whisper_context; lang_id: cint): whisper_token {.cdecl, importc.}

##  Task tokens
proc whisper_token_translate*(): whisper_token {.cdecl, importc.}
proc whisper_token_transcribe*(): whisper_token {.cdecl, importc.}

##  Performance information from the default state.
proc whisper_print_timings*(ctx: ptr whisper_context) {.cdecl, importc.}
proc whisper_reset_timings*(ctx: ptr whisper_context) {.cdecl, importc.}

##  Print system information
proc whisper_print_system_info*(): cstring {.cdecl, importc.}

##  NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see whisper_free_params()
proc whisper_full_default_params_by_ref*(strategy: whisper_sampling_strategy): ptr whisper_full_params {.cdecl, importc.}
proc whisper_full_default_params*(strategy: whisper_sampling_strategy): whisper_full_params {.cdecl, importc.}

##  Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
##  Not thread safe for same context
##  Uses the specified decoding strategy to obtain the text.
proc whisper_full*(ctx: ptr whisper_context; params: whisper_full_params;
                  samples: ptr cfloat; n_samples: cint): cint {.cdecl, importc.}
proc whisper_full_with_state*(ctx: ptr whisper_context; state: ptr whisper_state;
                             params: whisper_full_params; samples: ptr cfloat;
                             n_samples: cint): cint {.cdecl, importc.}

##  Split the input audio in chunks and process each chunk separately using whisper_full_with_state()
##  Result is stored in the default state of the context
##  Not thread safe if executed in parallel on the same context.
##  It seems this approach can offer some speedup in some cases.
##  However, the transcription accuracy can be worse at the beginning and end of each chunk.
proc whisper_full_parallel*(ctx: ptr whisper_context; params: whisper_full_params;
                           samples: ptr cfloat; n_samples: cint; n_processors: cint): cint {.cdecl, importc.}

##  Number of generated text segments
##  A segment can be a few words, a sentence, or even a paragraph.
proc whisper_full_n_segments*(ctx: ptr whisper_context): cint {.cdecl, importc.}
proc whisper_full_n_segments_from_state*(state: ptr whisper_state): cint {.cdecl, importc.}

##  Language id associated with the context's default state
proc whisper_full_lang_id*(ctx: ptr whisper_context): cint {.cdecl, importc.}

##  Language id associated with the provided state
proc whisper_full_lang_id_from_state*(state: ptr whisper_state): cint {.cdecl, importc.}

##  Get the start and end time of the specified segment
proc whisper_full_get_segment_t0*(ctx: ptr whisper_context; i_segment: cint): int64 {.cdecl, importc.}
proc whisper_full_get_segment_t0_from_state*(state: ptr whisper_state;
    i_segment: cint): int64 {.importc,header: "whisper.h".}
proc whisper_full_get_segment_t1*(ctx: ptr whisper_context; i_segment: cint): int64 {.cdecl, importc.}
proc whisper_full_get_segment_t1_from_state*(state: ptr whisper_state;
    i_segment: cint): int64 {.importc,header: "whisper.h".}

##  Get the text of the specified segment
proc whisper_full_get_segment_text*(ctx: ptr whisper_context; i_segment: cint): cstring {.cdecl, importc.}
proc whisper_full_get_segment_text_from_state*(state: ptr whisper_state;
    i_segment: cint): cstring {.importc,header: "whisper.h".}

##  Get number of tokens in the specified segment
proc whisper_full_n_tokens*(ctx: ptr whisper_context; i_segment: cint): cint {.cdecl, importc.}
proc whisper_full_n_tokens_from_state*(state: ptr whisper_state; i_segment: cint): cint {.cdecl, importc.}

##  Get the token text of the specified token in the specified segment
proc whisper_full_get_token_text*(ctx: ptr whisper_context; i_segment: cint;
                                 i_token: cint): cstring {.cdecl, importc.}
proc whisper_full_get_token_text_from_state*(ctx: ptr whisper_context;
    state: ptr whisper_state; i_segment: cint; i_token: cint): cstring {.cdecl, importc.}
proc whisper_full_get_token_id*(ctx: ptr whisper_context; i_segment: cint;
                               i_token: cint): whisper_token {.cdecl, importc.}
proc whisper_full_get_token_id_from_state*(state: ptr whisper_state;
    i_segment: cint; i_token: cint): whisper_token {.cdecl, importc.}

##  Get token data for the specified token in the specified segment
##  This contains probabilities, timestamps, etc.
proc whisper_full_get_token_data*(ctx: ptr whisper_context; i_segment: cint;
                                 i_token: cint): whisper_token_data {.cdecl, importc.}
proc whisper_full_get_token_data_from_state*(state: ptr whisper_state;
    i_segment: cint; i_token: cint): whisper_token_data {.cdecl, importc.}

##  Get the probability of the specified token in the specified segment
proc whisper_full_get_token_p*(ctx: ptr whisper_context; i_segment: cint;
                              i_token: cint): cfloat {.cdecl, importc.}
proc whisper_full_get_token_p_from_state*(state: ptr whisper_state; i_segment: cint;
    i_token: cint): cfloat {.cdecl, importc.}

##  Temporary helpers needed for exposing ggml interface
proc whisper_bench_memcpy*(n_threads: cint): cint {.cdecl, importc.}
proc whisper_bench_memcpy_str*(n_threads: cint): cstring {.cdecl, importc.}
proc whisper_bench_ggml_mul_mat*(n_threads: cint): cint {.cdecl, importc.}
proc whisper_bench_ggml_mul_mat_str*(n_threads: cint): cstring {.cdecl, importc.}
