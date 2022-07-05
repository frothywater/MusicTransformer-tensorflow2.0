from compile import main as compile
from preprocess import preprocess_midi_files_under as preprocess

if __name__ == "__main__":
    for dir in ["train", "valid", "test"]:
        preprocess(f"data/midi/{dir}", f"data/words/{dir}")
    compile()
