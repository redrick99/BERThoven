import json
import os
import pickle


def create_json_vocabulary(path_to_data: str, OCTAVE_AWARE=False):
    if OCTAVE_AWARE:
        in_path = os.path.join(path_to_data, "notes")
        out_path = os.path.join(path_to_data, "notes.txt")
    else:
        in_path = os.path.join(path_to_data, "notes_no_octave")
        out_path = os.path.join(path_to_data, "notes_no_octave.txt")

    with open(in_path, 'rb') as filepath:
        notes = pickle.load(filepath)
    jsonobj = json.loads(json.dumps(notes, default=str))

    with open(
            out_path,
            'w',
            encoding='utf-8'
    ) as outfile:
        json.dump(jsonobj, outfile, ensure_ascii=False, indent=4)


path = os.path.join(os.path.dirname(__file__), "resources", "data")
create_json_vocabulary(path)
