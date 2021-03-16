from pylsl import StreamInlet, resolve_stream

def mian():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)

if __name__ == "__main__":
    mian()