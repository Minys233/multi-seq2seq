from pathlib import Path
import shutil
import tempfile
import random


class FileSpliter:
    def __init__(self, from_fn, to_fn, split, name):
        self.location = Path(tempfile.mkdtemp())
        self.split = split
        self.name = name
        self.parts_from, self.total_from = self.split_file(from_fn, split, 'from')
        self.parts_to, self.total_to = self.split_file(to_fn, split, 'to')
        assert self.total_from == self.total_to, "from and to have different lines!"
        self.visited = [False for _ in range(split)]
        # create files: bigfile.1 ~ bigfile.split
        self.shuffle()

    def split_file(self, fname: str, fold, ext):
        fname = Path(fname)
        paths = [self.location/(self.name+f'.{ext}{i+1}') for i in range(fold)]
        fps = [open(self.location/(self.name+f'.{ext}{i+1}'), 'w') for i in range(fold)]
        lines = list(filter(lambda x: len(x)>1, open(fname).readlines()))
        assert len(lines) >= fold, f"this file is too small to split it into {fold} files"
        # random.shuffle(lines)
        portion = len(lines)//fold
        for idx, fp in enumerate(fps):
            for l in lines[portion*idx: portion*(idx+1)]:
                fp.write(l.strip()+'\n')
        for fp in fps:
            fp.close()
        return paths, len(lines)

    def shuffle(self):
        tmp = list(zip(self.parts_from, self.parts_to))
        random.shuffle(tmp)
        self.parts_from, self.parts_to = zip(*tmp)

    def __iter__(self):
        for i in range(self.split):
            yield self[i]

    def __getitem__(self, index):
        if all(self.visited):
            tmp = list(zip(self.parts_from, self.parts_to))
            random.shuffle(tmp)
            self.parts_from, self.parts_to = zip(*tmp)
            self.visited = [False for _ in self.visited]
        self.visited[index] = True
        return self.parts_from[index], self.parts_to[index]

    def __del__(self):
        shutil.rmtree(self.location)


