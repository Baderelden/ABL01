import hashlib
import hmac
import json
import os

import openai
import streamlit as st
from assessment import Assessment, PANEL_NAMES, assess, panel_summary, prepare_image

st.set_page_config(page_title='Vehicle damage assessment', page_icon='🚘', layout='wide')


def setting(name, default=''):
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


password = setting('APP_PASSWORD')
if password and not st.session_state.get('authenticated'):
    st.title('Vehicle assessment demo')
    with st.form('login'):
        entered = st.text_input('Demo password', type='password')
        submitted = st.form_submit_button('Open demo')
    if submitted:
        if hmac.compare_digest(entered.encode(), str(password).encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error('Incorrect password.')
    st.stop()

st.title('Vehicle damage assessment')
st.caption('Upload photos of one vehicle. Get a short assessment, a parts list and estimated repair labour.')

with st.sidebar:
    st.header('Assessment settings')
    model_options = setting('OPENAI_MODELS', ['gpt-4.1-mini', 'gpt-4.1', 'gpt-5-mini', 'gpt-5', 'gpt-6-astra'])
    if isinstance(model_options, str):
        model_options = [m.strip() for m in model_options.split(',') if m.strip()]
    if not model_options:
        model_options = ['gpt-4.1-mini', 'gpt-4.1', 'gpt-5-mini', 'gpt-5', 'gpt-6-astra']
    model = st.selectbox('LLM engine', model_options)
    st.caption('Model access depends on your OpenAI API account.')
    api_key = setting('OPENAI_API_KEY')
    if not api_key:
        api_key = st.text_input('OpenAI API key', type='password', help='Used for this session only.')
    notes = st.text_area('Vehicle details (optional)', max_chars=1500,
                         placeholder='For example: electric SUV, model/year, or large truck.')
    st.caption('Photos are sent to OpenAI when you click Assess vehicle.')
    if st.button('Clear photos and assessment'):
        st.session_state.pop('report', None)
        st.session_state.upload_generation = st.session_state.get('upload_generation', 0) + 1
        st.rerun()

files = st.file_uploader('Upload vehicle photos', type=['jpg', 'jpeg', 'png', 'webp'],
                         accept_multiple_files=True,
                         key=f"photos_{st.session_state.get('upload_generation', 0)}")
st.caption('Up to 12 photos • 10 MB per photo • 40 MB total • JPEG, PNG or WebP')
images, errors = [], []
if len(files) > 12:
    errors.append('Please upload no more than 12 photos.')
elif sum(f.size for f in files) > 40 * 1024 * 1024:
    errors.append('Please keep the total upload under 40 MB.')
else:
    columns = st.columns(4)
    for i, f in enumerate(files):
        try:
            data = prepare_image(f.getvalue())
            images.append(data)
            columns[i % 4].image(data, caption=f'Photo {i + 1}: {f.name}', use_container_width=True)
        except ValueError as exc:
            errors.append(f'{f.name}: {exc}')
for error in errors:
    st.error(error)

# A report is tied to the images, model and notes that produced it.
fingerprint = hashlib.sha256()
for f in files:
    fingerprint.update(hashlib.sha256(f.getvalue()).digest())
fingerprint.update(json.dumps([model, notes, "visual-repair-size-v2"]).encode())
case_id = fingerprint.hexdigest()
if st.session_state.get('report', {}).get('case_id') != case_id:
    st.session_state.pop('report', None)

if st.button('Assess vehicle', type='primary', disabled=not images or bool(errors)):
    st.session_state.pop('report', None)
    if not api_key:
        st.error('Add an OpenAI API key in the sidebar or in Streamlit secrets.')
    else:
        try:
            with st.spinner('Reviewing all vehicle photos…'):
                with openai.OpenAI(api_key=api_key, timeout=120, max_retries=1) as client:
                    result, usage = assess(client, model, images, notes)
            st.session_state.report = {
                'case_id': case_id, 'model': model,
                'assessment': result.model_dump(mode='json'), 'usage': usage,
            }
        except openai.AuthenticationError:
            st.error('The API key was not accepted. Check your OpenAI API key.')
        except openai.RateLimitError:
            st.error('API quota or rate limit reached. Check billing/limits, then retry.')
        except (openai.NotFoundError, openai.PermissionDeniedError):
            st.error('This model is unavailable to your API account. Choose another model.')
        except openai.BadRequestError:
            st.error('The API rejected this request. Check that the selected model supports images and structured outputs.')
        except openai.APIConnectionError:
            st.error('Could not connect to OpenAI, or the request timed out. Please retry.')
        except openai.APIError:
            st.error('OpenAI could not complete the assessment. Please try again later.')
        except ValueError:
            st.error('No valid, complete assessment was returned. Try clearer photos or another model.')

if 'report' in st.session_state:
    report = st.session_state.report
    result = Assessment.model_validate(report['assessment'])
    count, size = panel_summary(result)
    st.divider()
    st.subheader('Assessment')
    st.write(result.short_description)
    if result.status != 'Assessable':
        st.warning(result.status)
    low, high = result.labour.minimum_hours, result.labour.maximum_hours
    hours = f'{low:g}–{high:g} h' if low is not None else 'Not estimable'
    a, b, c = st.columns(3)
    a.metric('Repair size (visual estimate)', size)
    b.metric('Confirmed damaged panels', count)
    c.metric('Estimated labour', hours)
    st.caption('Based on visible severity, deformation and likely repair complexity. Panel count is descriptive only.')
    st.write('Why this size: ' + result.repair_size.explanation)
    st.caption('Size confidence: ' + result.repair_size.confidence + ' · Evidence photos: ' +
               (', '.join(map(str, result.repair_size.photo_numbers)) or 'None'))
    st.write(f'Vehicle: {result.vehicle_type} · Powertrain: {result.powertrain}')
    st.subheader('Parts and damage')
    if result.damaged_parts:
        st.dataframe([
            {'Part': p.part.value, 'Damage': p.damage, 'Finding': p.certainty,
             'Photo(s)': ', '.join(map(str, p.photo_numbers)),
             'Panel counted': 'Yes' if p.part.value in PANEL_NAMES and p.certainty == 'Observed' else 'No'}
            for p in result.damaged_parts
        ], hide_index=True, use_container_width=True)
    else:
        st.info('No damaged parts could be confirmed from these photos.')
    st.write('Labour assumptions: ' + result.labour.basis_and_assumptions)
    st.subheader('Special needs')
    if result.special_needs:
        for need in result.special_needs:
            st.write(f'• {need.requirement} — {need.reason} ({need.basis})')
    else:
        st.write('None identified from the available evidence; vehicle details may still need confirmation.')
    if result.limitations or result.additional_photos_needed:
        with st.expander('Limitations and useful additional photos', expanded=True):
            for item in result.limitations:
                st.write('• ' + item)
            for item in result.additional_photos_needed:
                st.write('• Additional photo: ' + item)
    payload = dict(report, panel_count=count, damage_size=size, sizing_method='visual_severity_and_complexity_v2')
    payload.pop('case_id')
    st.download_button('Download assessment (JSON)', json.dumps(payload, indent=2, ensure_ascii=False),
                       file_name='vehicle-assessment.json', mime='application/json')
    with st.expander('Model and usage'):
        st.write(report['model'])
        st.json(report['usage'] or {})

with st.expander('How the assessment works'):
    st.write('The selected model receives all photos together. A fixed part vocabulary keeps names consistent. '
             'The app counts unique, observed body panels; possible findings, lamps, glass, wheels and other '
             'components do not increase the panel count. Bumper covers are counted as panels in this demo.')
    st.write('Repair size is assessed separately from the images: Small means minor cosmetic work; '
             'Medium means localised damage needing substantive repair; Large means severe deformation, '
             'extensive damage or complex repair. These are demo categories, not an industry standard.')
    st.write('Labour hours are rough photo-based estimates, not a repair quotation or elapsed days. '
             'A repairer must inspect hidden damage and confirm the work required.')
    st.write('Images are resized to a maximum 1,600 pixels and re-encoded without EXIF metadata. '
             'The app does not write photos to disk. Requests use store=False; this is not a zero-retention guarantee.')
