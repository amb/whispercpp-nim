# nim --cc:vcc c -r -d:release nim_test.nim

import whisper
import wavfile

var pcmf32 = loadWav("samples/jfk.wav").toFloat()
var ctx*: ptr whisper_context = whisper_init_from_file("models/ggml-base.bin")

echo("Init done.")

var wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
wparams.print_progress = false
if whisper_full(ctx, wparams, cast[ptr cfloat](pcmf32), pcmf32.len.cint) != 0:
    echo("Failed to process audio")
    quit(QuitFailure)

for i in 0..<whisper_full_n_segments(ctx):
    echo(whisper_full_get_segment_text(ctx, i.cint))

whisper_free(ctx)
