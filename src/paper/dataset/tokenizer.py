import re
import sys
from typing import Generator


class Tokenizer(object):
    REGEX_ATOMS = re.compile(
        "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    @classmethod
    def build(cls, signature: str) -> Generator:
        token_specification = [
            ("FINGERPRINT", r"(\d+,)*"),
            ("ATOMS", r"[\w=:\d\[\]\(\)]+"),
            (
                "BOUND",
                r"UNSPECIFIED|SINGLE|DOUBLE|TRIPLEQUADRUPLE|QUINTUPLE|HEXTUPLE|ONEANDAHALF|TWOANDAHALF|THREEANDAHALF|FOURANDAHALF|FIVEANDAHALF|AROMATI|IONIC|HYDROGEN|THREECENTER|DATIVEONE|DATIVE|DATIVEL|DATIVER|OTHER|ZERO",
            ),
            ("SPACER", r"[\s\.\|]"),
        ]
        tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
        is_fingerprint = False
        for mo in re.finditer(tok_regex, signature):
            kind = mo.lastgroup
            value = mo.group()
            tokens = []

            if value == "":
                continue
            if kind == "FINGERPRINT":
                tokens = list(value)
            elif kind == "ATOMS":
                tokens = [token for token in cls.REGEX_ATOMS.findall(value)]
            elif kind == "BOUND":
                tokens = value  # list(mo.group(1, 2, 3))
            elif kind == "SPACER":
                if value == " ":
                    tokens = ["!"]
                else:
                    tokens = value
            yield tokens

    @classmethod
    def tokenize(cls, signature: str, sep: str = " ") -> str:
        res = []
        for token in cls.build(signature=signature):
            res.extend(token)
        return sep.join(res)


if __name__ == "__main__":
    signature = "1,8,2040,C=[O:1].DOUBLE|2,6,1021,C[C:1](C)=O 2,6,1021,C[C:1](C)=O.DOUBLE|1,8,2040,C=[O:1].SINGLE|2,6,1439,C[CH:1](O)O.SINGLE|2,6,1928,C[CH:1](C)O"
    print(Tokenizer.tokenize(signature=signature))
