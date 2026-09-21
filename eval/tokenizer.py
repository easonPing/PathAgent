"""COCO PTB settings with temporary files outside the read-only Python environment."""
import subprocess
import tempfile
from pathlib import Path


class PTBTokenizer:
    def tokenize(self, captions):
        from pycocoevalcap.tokenizer import ptbtokenizer
        jar = Path(ptbtokenizer.__file__).parent / ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR
        identities = [key for key, values in captions.items() for _ in values]
        sentences = '\n'.join(value['caption'].replace('\n', ' ')
                              for values in captions.values() for value in values)
        with tempfile.TemporaryDirectory(prefix='pathagent-ptb-') as directory:
            path = Path(directory) / 'captions.txt'
            path.write_text(sentences, encoding='utf-8')
            result = subprocess.run(['java', '-cp', str(jar), 'edu.stanford.nlp.process.PTBTokenizer',
                                     '-preserveLines', '-lowerCase', str(path)],
                                    check=True, capture_output=True, text=True, timeout=120)
        lines = result.stdout.split('\n')
        if len(lines) < len(identities):
            raise RuntimeError('PTB tokenizer returned fewer lines than samples')
        output = {}
        for key, line in zip(identities, lines):
            tokens = ' '.join(word for word in line.rstrip().split(' ')
                              if word not in ptbtokenizer.PUNCTUATIONS)
            output.setdefault(key, []).append(tokens)
        return output
