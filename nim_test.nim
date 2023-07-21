# nim --cc:vcc c -r -d:danger -d:strip --mm:arc nim_test.nim models/ggml-base.bin samples/jfk.wav

import whisper
import wavfile
import std/os

# Read command-line parameters
var args = commandLineParams()
echo args
if args.len != 2:
    echo("Usage: nim_test <model.bin> <input.wav>")
    quit(QuitFailure)

# Load input file
var pcmf32 = loadWav(args[1]).toFloat(16000)
var ctx*: ptr whisper_context = whisper_init_from_file(args[0])

var wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
wparams.print_progress = false
if whisper_full(ctx, wparams, cast[ptr cfloat](pcmf32[0].addr), pcmf32.len.cint) != 0:
    echo("Failed to process audio")
    quit(QuitFailure)

for i in 0..<whisper_full_n_segments(ctx):
    echo(whisper_full_get_segment_text(ctx, i.cint))

whisper_free(ctx)
