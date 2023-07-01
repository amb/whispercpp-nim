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


proc toFloat*(wav: WavFile): seq[float32] =
    # TODO: convert every wave file to mono, 16 kHz (for Whisper)
    assert wav.bits == 16
    assert wav.channels == 1
    var rseq: seq[float32] = @[]
    var arr = cast[ptr UncheckedArray[int16]](wav.data[0].unsafeAddr)
    # TODO: inprecise
    let mpl = 1.0/32000.0
    for i in 0..<wav.size div 2:
        rseq.add(float32(arr[i])*mpl)
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
