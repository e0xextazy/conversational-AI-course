import os
from typing import List


class DirectoryReader:
    def __init__(self, input_dir: str) -> None:
        self.input_dir = input_dir

    def load_data(self) -> List[str]:
        documents = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.input_dir, filename), 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        return documents


def get_data(input_dir: str) -> List[str]:
    reader = DirectoryReader(input_dir)
    txt_docs = reader.load_data()
    return txt_docs
