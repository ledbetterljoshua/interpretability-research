"""Native formats for unmodified references; no model or tokenizer loads.

The completion adapter deliberately renders the existing question/example body
without chat framing. All encoding, padding and vocabulary operations still use
the reference's own tokenizer. This lets the same instrument and training code
exercise both formats without modifying already executed research sources.
"""
REFERENCES = {
    "post": dict(model="Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                 format="chat", answer_strings=list("ABCD")),
    "base": dict(model="Qwen/Qwen3-1.7B-Base", revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                 format="completion", answer_strings=[" "+c for c in "ABCD"]),
}


class ReferenceTokenizer:
    def __init__(self, tokenizer, reference):
        self.native = tokenizer
        self.reference = reference
        self.spec = REFERENCES[reference]

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, *args, **kwargs):
        return self.native(*args, **kwargs)

    @property
    def padding_side(self):
        return self.native.padding_side

    @padding_side.setter
    def padding_side(self, value):
        self.native.padding_side = value

    def apply_chat_template(self, messages, **kwargs):
        if self.spec["format"] == "chat":
            return self.native.apply_chat_template(messages, **kwargs)
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == (
            "Answer the multiple-choice science question. Reply with exactly one letter: A, B, C, or D.")
        content = messages[1]["content"]
        assert content.endswith("\nAnswer:")
        return content

    def choice_ids(self):
        values = [self.native.encode(s, add_special_tokens=False) for s in self.spec["answer_strings"]]
        assert all(len(v) == 1 for v in values)
        ids = [v[0] for v in values]
        assert len(set(ids)) == 4
        return ids
