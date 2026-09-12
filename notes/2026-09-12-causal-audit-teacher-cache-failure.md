# Retained teacher preflight cache failure

`weak-teacher-v1` stopped before model loading. Its call to offline
`snapshot_download` omitted the inference-file filter used during download, so
Hugging Face correctly reported an incomplete full-repository snapshot. The
eight intended inference files had downloaded successfully. No predictions,
training updates or eligibility results were produced.

The original `teacher.py` and failed manifest remain unchanged. Before retrying,
`teacher_v2.py` restricts the offline completeness check to the same inference
patterns as the download: weights, configuration, tokenizer, vocabulary, merges,
special tokens and generation configuration. The original committed teacher
plan, model revision, populations, forecasts and resource limits are unchanged.
The retry writes a distinct `weak-teacher-v2` directory. Commit this correction
before that script loads a model.
