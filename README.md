# Climb Route Grader

Upload a photo of a bouldering wall → get a predicted V-grade.

Photo-in / grade-out, built on the two-stage plan in `DESIGN.md`
(perception → grading). This is the **first page**: a runnable Streamlit UI
with a *stub* grader so the whole flow works before the PyTorch model exists.

## Run

```bash
cd climb-grader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Layout

```
app/
  main.py     # the first page: upload → grade UI
  grader.py   # predictor interface + STUB. Real model plugs in here.
```

The stub hashes the image to a reproducible pseudo-grade, so the same photo
always returns the same result. Replace the body of `grader.predict` with the
real perception → grading pipeline; the UI stays unchanged.
