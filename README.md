# Vehicle damage assessment demo

A Streamlit app using the OpenAI Responses API directly. No LangChain required.

## Included features

- LLM dropdown, including GPT-4.1 mini, GPT-4.1, GPT-5 mini, GPT-5 and GPT-6 Astra; configurable through secrets.
- Multiple JPEG, PNG and WebP uploads with a four-column thumbnail gallery.
- All photographs sent together for one vehicle assessment.
- Very short description, standardised parts and damage table, evidence photo numbers.
- Unique observed body-panel count: Small = 1–2, Medium = 3–5, Large = 6+.
- Estimated active repair labour range, assumptions and relevant special needs.
- Possible damage distinguished from observed damage; insufficient evidence handled explicitly.
- Downloadable JSON report, session-only report storage and optional shared demo password.

## Run locally

Use Python 3.11 or 3.12. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enter an OpenAI API key in the sidebar, or copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml` and replace the placeholder. Never commit the real secrets file.
An `OPENAI_API_KEY` environment variable also works. No key is bundled.

## Deploy on Streamlit Community Cloud

1. Unzip the project. Put its contents in a GitHub repository, with `app.py`,
   `assessment.py` and `requirements.txt` at the repository root. Include
   `.streamlit/config.toml` and `.gitignore`; exclude `.venv`, caches and real secrets.
2. Open https://share.streamlit.io and connect the repository.
3. Create an app, select the repository and branch, and set the main file to `app.py`.
4. In Advanced settings, choose Python 3.11 or 3.12 and add these TOML secrets:

```toml
OPENAI_API_KEY = "your-real-openai-api-key"
OPENAI_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5", "gpt-6-astra"]
APP_PASSWORD = "choose-a-long-random-demo-password"
```

5. Deploy. Open the app, enter the demo password, upload vehicle photos and click
   **Assess vehicle**. If you omit the server API key, users can enter their own
   key in the sidebar. The password is optional, but useful when your key funds requests.

Streamlit's instructions for keeping secrets out of GitHub:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

This package is prepared for deployment; it has not been published to your account.

## Model configuration

The initial choices support image inputs, Responses and structured outputs.
Account permissions and availability can differ. To add other models, update
`OPENAI_MODELS` with exact API model IDs supporting both images and structured
outputs. This is a configured allowlist, not automatic model discovery.
GPT-5 mini, GPT-5 and GPT-6 Astra receive a 16,000-token output budget to
accommodate reasoning and the visible report. GPT-5 and GPT-6 Astra use low
reasoning effort. Other models may need their own settings in `assessment.py`.
If an existing deployment sets OPENAI_MODELS in secrets, update that list too;
it overrides the code defaults.

Model documentation:
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://developers.openai.com/api/docs/models/gpt-5
- https://developers.openai.com/api/docs/models/gpt-5-mini
- https://developers.openai.com/api/docs/models/gpt-4.1-mini
- https://developers.openai.com/api/docs/models/gpt-4.1

## Counting rules and scope

The fixed part vocabulary lives in `assessment.py`. It uses familiar British body
repair names, not an insurer/OEM parts coding standard. Left/right is the vehicle's
own orientation. Unknown part identity stays unclassified instead of guessing.

The default panel list includes bumper covers, bonnet, roof, boot lid/tailgate,
wings, rear quarter panels, doors and sills. Glass, lamps, mirrors, wheels and
other components are reported but excluded from the panel count. Each observed
panel counts once across all photos; possible damage is excluded. Zero confirmed
panels displays **No confirmed panel damage**, not Small. Unassessable inputs
show **Not assessable**. Hidden/unseen or unidentified panels cannot be counted,
so the category reflects only confirmed visible panels, not the entire repair.

The standard list primarily covers passenger vehicles. Special truck body panels
use Other / unidentified part; extend the vocabulary before using this for detailed
commercial-body assessments. Truck handling requirements can still be reported.

Hours are an LLM estimate of active labour, not elapsed days, manufacturer labour
book times or a guaranteed quotation. The model can return no estimate if evidence
is insufficient. Panel count measures extent, not severity. A single panel can
still have serious damage. Inspection is needed before repair or safety decisions.

Electric/hybrid status should be supplied in the optional vehicle details when
known. The model must label uncertain requirements rather than invent a powertrain.

## Implementation and data handling

`client.responses.parse(..., text_format=Assessment)` validates structured output
with Pydantic. All images are included in one request. The Python code computes
the category rather than relying on model arithmetic. Reports are invalidated
when photos, notes or model selection change.

Photos are re-encoded without EXIF, resized to at most 1,600 pixels, and sent as
base64 data URLs with high detail. The demo accepts up to 12 images, 10 MB each,
40 MB total and 25 megapixels per image. Smaller images reduce request size but
may lose fine damage detail; upload close-ups when needed.

The app does not write uploaded images to disk or a database. Uploads and results
remain in Streamlit session memory; Clear photos and assessment removes the app's
current references. This is not a promise of secure memory erasure. Requests set
`store=False`; this does not establish zero retention or data compliance by itself.
The demo password is a shared access gate, not full user management or rate limiting.

API guides used:
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/structured-outputs

## Verification

Run the included offline checks:

```bash
python -m unittest -v
```

Checks cover threshold boundaries, duplicate panels, excluding possible/non-panel
damage, invalid hour ranges, image conversion, actual SDK parsing with a mocked
HTTP response and refusals. The initial Streamlit screen was also smoke-tested.
The GPT-5/GPT-6 update passed the expanded mocked SDK checks and dropdown
smoke test. No live API call or real-photo accuracy evaluation was performed. Validate against
repairer-labelled cases before using output for operational estimates.

## Files

- `app.py`: user interface, key configuration, result display and downloads.
- `assessment.py`: part vocabulary, schema, prompt, image preparation and API call.
- `requirements.txt`: supported dependency ranges.
- `requirements-tested.txt`: exact direct-package versions used in offline checks.
- `test_assessment.py`: offline checks; no API key required.
- `.streamlit/config.toml`: theme and upload limit.
- `.streamlit/secrets.toml.example`: placeholder configuration only.
