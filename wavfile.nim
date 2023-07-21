import streams, std/base64, std/strformat

type
    WavFile* = object
        data*: seq[uint8]
        size*: int
        freq*: int
        bits*: int
        channels*: int

    wavHeaderObj* = object
        ChunkID*: array[4, char]
        ChunkSize: uint32
        Format: array[4, char]
        FmtChunkID: array[4, char]
        FmtChunkSize: uint32
        AudioFormat: uint16
        NumChannels: uint16
        SampleRate: uint32
        ByteRate: uint32
        BlockAlign: uint16
        BitsPerSample: uint16

    wavChunkObj* = object
        DataChunkID: string
        DataChunkSize*: uint32
        Data: string


proc readDataChunk(f: FileStream): wavChunkObj =
    var chunk = wavChunkObj()
    chunk.DataChunkID = f.readStr(4)
    chunk.DataChunkSize = f.readUint32()
    chunk.Data = f.readStr(chunk.DataChunkSize.int)
    return chunk


proc toFloat*(wav: WavFile, outputFreq: int): seq[float32] =
    # Convert wav file to float32 sequence
    assert wav.bits == 16
    var inputFreq: int = wav.freq

    var input: seq[float32] = @[]
    var arr = cast[ptr UncheckedArray[int16]](wav.data[0].unsafeAddr)

    let mpl: float32 = 1.0/32768.0f
    if wav.channels == 1:
        for i in 0..<wav.size div 2:
            input.add(float32(arr[i])*mpl)
    elif wav.channels == 2:
        for i in 0..<wav.size div 4:
            input.add((float32(arr[i*2]) + float32(arr[i*2+1])) * mpl * 0.5)
    else:
        doAssert false, "Unsupported number of channels"

    # Equal input and output frequency, return input
    if inputFreq == outputFreq:
        return input

    # Upsample or downsample to match output frequency
    var rseq: seq[float32] = @[]
    var step: float32 = float32(inputFreq) / float32(outputFreq)
    var index: float32

    doAssert inputFreq > 0
    doAssert outputFreq > 0
    doAssert inputFreq < 200000
    doAssert outputFreq < 200000
    
    if step > 1.0:
        # downsample
        index = 0.0
        var accumulator: float32 = 0.0
        var count: float32 = 0.0

        for sample in input:
            accumulator += sample
            count += 1.0

            if count >= step:
                rseq.add(accumulator / count)
                accumulator = 0.0
                count -= step
    else:
        # upsample
        index = 1.0
        while index < input.len.float:
            var a = input[(index-1.0f).int]
            var b = input[index.int]
            var ratio = index - index.int.float32
            rseq.add(a + (b-a)*ratio)
            index += step

    return rseq


proc loadWav*(filePath: string): WavFile =
    # Load PCM data from wav file.
    var f = newFileStream(filePath)

    var header = wavHeaderObj()
    var count = f.readData(addr(header), sizeof(wavHeaderObj))

    assert count == sizeof(wavHeaderObj)
    assert header.ChunkID == "RIFF"
    assert header.Format == "WAVE"
    assert header.FmtChunkID == "fmt "
    assert header.AudioFormat == 1

    var chunk = wavChunkObj()
    while chunk.DataChunkID != "data":
        chunk = f.readDataChunk()

    result.channels = int header.NumChannels
    result.size = chunk.Data.len
    result.freq = int header.SampleRate
    result.bits = int header.BitsPerSample
    result.data = cast[seq[uint8]](chunk.Data)
