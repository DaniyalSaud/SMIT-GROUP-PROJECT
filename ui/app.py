import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv("FLOOD_API_URL", "http://127.0.0.1:8000/predict")


FEATURE_SECTIONS = {
	"Hydrology": [
		"MonsoonIntensity",
		"TopographyDrainage",
		"RiverManagement",
		"DamsQuality",
		"Siltation",
		"Watersheds",
	],
	"Land Use": [
		"Deforestation",
		"Urbanization",
		"AgriculturalPractices",
		"Encroachments",
		"CoastalVulnerability",
		"Landslides",
	],
	"Preparedness": [
		"ClimateChange",
		"IneffectiveDisasterPreparedness",
		"DrainageSystems",
	],
	"Infrastructure": [
		"DeterioratingInfrastructure",
		"WetlandLoss",
	],
}


st.set_page_config(
	page_title="Flood Prediction Dashboard",
	page_icon="🌊",
	layout="wide",
)


FEATURES = [
	"MonsoonIntensity",
	"TopographyDrainage",
	"RiverManagement",
	"Deforestation",
	"Urbanization",
	"ClimateChange",
	"DamsQuality",
	"Siltation",
	"AgriculturalPractices",
	"Encroachments",
	"IneffectiveDisasterPreparedness",
	"DrainageSystems",
	"CoastalVulnerability",
	"Landslides",
	"Watersheds",
	"DeterioratingInfrastructure",
	"WetlandLoss",
]

SAMPLE_INPUT = {
	"MonsoonIntensity": 1,
	"TopographyDrainage": 3,
	"RiverManagement": 4,
	"Deforestation": 5,
	"Urbanization": 3,
	"ClimateChange": 2,
	"DamsQuality": 8,
	"Siltation": 9,
	"AgriculturalPractices": 3,
	"Encroachments": 2,
	"IneffectiveDisasterPreparedness": 9,
	"DrainageSystems": 8,
	"CoastalVulnerability": 6,
	"Landslides": 2,
	"Watersheds": 1,
	"DeterioratingInfrastructure": 1,
	"WetlandLoss": 4,
}


def build_payload() -> dict[str, float]:
	return {feature: float(st.session_state.get(feature, SAMPLE_INPUT[feature])) for feature in FEATURES}


def risk_label(score: float) -> str:
	if score < 0.33:
		return "Low"
	if score < 0.66:
		return "Moderate"
	return "High"


def risk_message(score: float) -> str:
	if score < 0.33:
		return "Conditions look relatively stable, but continue monitoring the dominant drivers."
	if score < 0.66:
		return "Risk is rising. Review drainage, land-use pressure, and preparedness together."
	return "The signal is elevated. Prioritize mitigation planning and local response readiness."


for feature in FEATURES:
	st.session_state.setdefault(feature, float(SAMPLE_INPUT[feature]))

st.session_state.setdefault("prediction_response", None)
st.session_state.setdefault("prediction_error", None)


st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Source+Code+Pro:wght@400;600&display=swap');

	.stApp {
		background:
			radial-gradient(1000px 700px at 10% 0%, rgba(70, 129, 158, 0.16) 0%, transparent 55%),
			radial-gradient(900px 600px at 95% 5%, rgba(232, 170, 84, 0.12) 0%, transparent 52%),
			linear-gradient(180deg, #f7fbfd 0%, #f4f7f9 100%);
		color: #12212b;
	}

	h1, h2, h3, h4, h5, h6 {
		font-family: 'Space Grotesk', sans-serif;
		letter-spacing: -0.02em;
		color: #0f2533;
	}

	p, span, div, label, input, textarea {
		font-family: 'Space Grotesk', sans-serif;
	}

	.hero {
		background: linear-gradient(135deg, #0f2d3d 0%, #12465c 45%, #2f6d84 100%);
		color: #f3f7f9;
		padding: 28px 28px 24px 28px;
		border-radius: 22px;
		box-shadow: 0 22px 54px rgba(13, 33, 43, 0.24);
		border: 1px solid rgba(255, 255, 255, 0.12);
	}

	.hero-kicker {
		font-size: 0.95rem;
		letter-spacing: 0.2em;
		text-transform: uppercase;
		opacity: 0.75;
		margin-bottom: 12px;
	}

	.hero-title {
		font-size: 2.55rem;
		font-weight: 700;
		margin-bottom: 8px;
	}

	.hero-subtitle {
		font-size: 1.05rem;
		max-width: 720px;
		opacity: 0.9;
		line-height: 1.55;
	}

	.card {
		background: #ffffff;
		border-radius: 18px;
		padding: 18px 20px;
		border: 1px solid rgba(15, 45, 61, 0.08);
		box-shadow: 0 12px 24px rgba(12, 24, 32, 0.08);
		min-height: 118px;
	}

	.card-title {
		font-size: 0.9rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: #546371;
		margin-bottom: 8px;
	}

	.card-value {
		font-size: 1.45rem;
		font-weight: 700;
		color: #0f2d3d;
		line-height: 1.1;
	}

	.panel {
		background: rgba(255, 255, 255, 0.84);
		border: 1px solid rgba(15, 45, 61, 0.08);
		border-radius: 20px;
		padding: 18px 18px 10px 18px;
		margin-bottom: 18px;
		box-shadow: 0 14px 30px rgba(15, 37, 51, 0.06);
	}

	.tag {
		display: inline-block;
		padding: 6px 10px;
		border-radius: 999px;
		background: #e6f1f7;
		color: #0f2d3d;
		font-size: 0.8rem;
		font-weight: 600;
		margin-right: 6px;
	}

	.subtle {
		color: #60727e;
	}

	.result-box {
		padding: 18px;
		border-radius: 18px;
		background: linear-gradient(180deg, #f7fbfd 0%, #eef5f9 100%);
		border: 1px solid rgba(15, 45, 61, 0.08);
	}

	.result-score {
		font-size: 2rem;
		font-weight: 700;
		color: #0f2d3d;
	}

	.result-label {
		font-size: 0.82rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		font-weight: 700;
		color: #60727e;
		margin-bottom: 6px;
	}

	.sidebar-chip {
		display: block;
		padding: 10px 12px;
		border-radius: 12px;
		background: #f1f7fa;
		border: 1px solid rgba(15, 45, 61, 0.08);
		margin-bottom: 10px;
		font-size: 0.95rem;
	}

	.section-heading {
		font-size: 0.85rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		font-weight: 700;
		color: #60727e;
		margin-bottom: 10px;
	}

	.feature-grid {
		background: #ffffff;
		border-radius: 18px;
		padding: 12px 14px 6px 14px;
		border: 1px solid rgba(15, 45, 61, 0.08);
	}

	.mono {
		font-family: 'Source Code Pro', monospace;
	}

	div[data-testid="stForm"] {
		background: transparent;
		border: 0;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


with st.sidebar:
	st.markdown("<div class='sidebar-chip'><strong>Endpoint</strong><br><span class='subtle'>" + API_URL + "</span></div>", unsafe_allow_html=True)
	st.markdown("<div class='sidebar-chip'><strong>Feature count</strong><br><span class='subtle'>17 model inputs</span></div>", unsafe_allow_html=True)
	st.markdown("<div class='sidebar-chip'><strong>Model shape</strong><br><span class='subtle'>JSON → Spark → MLflow</span></div>", unsafe_allow_html=True)
	st.caption("Use the buttons below to reset the form or load the sample configuration.")
	if st.button("Load sample values", use_container_width=True):
		for feature in FEATURES:
			st.session_state[feature] = float(SAMPLE_INPUT[feature])
		st.session_state.pop("prediction_response", None)
		st.session_state.pop("prediction_error", None)
		st.rerun()
	if st.button("Clear result", use_container_width=True):
		st.session_state.pop("prediction_response", None)
		st.session_state.pop("prediction_error", None)
		st.rerun()
	st.markdown("---")
	st.write("**Deployment notes**")
	st.write("The UI posts to the FastAPI backend at the configured endpoint and renders the response in place.")


st.markdown(
	"""
	<div class="hero">
		<div class="hero-kicker">Flood Prediction Dashboard</div>
		<div class="hero-title">Assess flood pressure from terrain, climate, and resilience signals</div>
		<div class="hero-subtitle">
			A professional control surface for the 17-feature flood model. Adjust the drivers,
			submit a request, and review the response without leaving the page.
		</div>
	</div>
	""",
	unsafe_allow_html=True,
)

st.write("")


col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
	st.markdown(
		"""
		<div class="card">
			<div class="card-title">Target</div>
			<div class="card-value">FloodProbability</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

with col_b:
	st.markdown(
		"""
		<div class="card">
			<div class="card-title">Features</div>
			<div class="card-value">17 signals</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

with col_c:
	st.markdown(
		"""
		<div class="card">
			<div class="card-title">Pipeline</div>
			<div class="card-value">Vector + Scaling</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

with col_d:
	st.markdown(
		"""
		<div class="card">
			<div class="card-title">Status</div>
			<div class="card-value">Live API</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


st.write("")

left, right = st.columns([1.3, 0.9])

with left:
	st.markdown("<div class='panel'>", unsafe_allow_html=True)
	st.markdown("<div class='section-heading'>Prediction inputs</div>", unsafe_allow_html=True)
	st.write("Edit the model drivers below, then submit the form to request a live prediction.")

	with st.form("prediction_form", clear_on_submit=False):
		for section_name, section_features in FEATURE_SECTIONS.items():
			st.markdown(f"<div class='feature-grid'>", unsafe_allow_html=True)
			st.markdown(f"<div class='section-heading'>{section_name}</div>", unsafe_allow_html=True)
			section_columns = st.columns(2)
			for index, feature in enumerate(section_features):
				with section_columns[index % 2]:
					st.slider(
						feature,
						min_value=0.0,
						max_value=10.0,
						value=float(st.session_state[feature]),
						step=0.1,
						key=feature,
					)
			st.markdown("</div>", unsafe_allow_html=True)

		submit_col, helper_col = st.columns([0.42, 0.58])
		with submit_col:
			submit = st.form_submit_button("Run prediction", use_container_width=True)
		with helper_col:
			st.caption("The backend expects a JSON body and returns a single flood probability value.")

	if submit:
		input_values = build_payload()
		try:
			request = Request(
				API_URL,
				data=json.dumps(input_values).encode("utf-8"),
				headers={"Content-Type": "application/json"},
				method="POST",
			)
			with st.spinner("Contacting FastAPI backend..."):
				with urlopen(request, timeout=60) as response:
					st.session_state["prediction_response"] = json.loads(response.read().decode("utf-8"))
					st.session_state.pop("prediction_error", None)
		except HTTPError as error:
			st.session_state["prediction_error"] = error.read().decode("utf-8")
			st.session_state.pop("prediction_response", None)
		except URLError as error:
			st.session_state["prediction_error"] = str(error)
			st.session_state.pop("prediction_response", None)

	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("<div class='panel'>", unsafe_allow_html=True)
	st.markdown("<div class='section-heading'>Prepared payload</div>", unsafe_allow_html=True)
	st.code(build_payload(), language="python")
	st.caption(f"Payload posted to {API_URL}.")
	st.markdown("</div>", unsafe_allow_html=True)

with right:
	st.markdown("<div class='panel'>", unsafe_allow_html=True)
	st.markdown("<div class='section-heading'>Prediction result</div>", unsafe_allow_html=True)
	response = st.session_state.get("prediction_response")
	error = st.session_state.get("prediction_error")

	if response:
		score = float(response["predicted_flood_probability"])
		st.markdown(
			f"""
			<div class='result-box'>
				<div class='result-label'>Flood risk</div>
				<div class='result-score'>{score:.4f}</div>
				<div class='subtle'>{risk_label(score)} risk</div>
				<p>{risk_message(score)}</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
		st.metric("Status", response["status"].title())
		st.metric("Model", response["model_name"])
		st.metric("Pipeline", response["pipeline_name"])
	elif error:
		st.error(error)
	else:
		st.info("Submit the form to see the live backend response here.")
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("<div class='panel'>", unsafe_allow_html=True)
	st.markdown("<div class='section-heading'>Model context</div>", unsafe_allow_html=True)
	st.write("The model uses the flood feature contract defined in params.yaml and the FastAPI backend handles inference.")
	st.markdown(
		"""
		<span class="tag">Hydrology</span>
		<span class="tag">Infrastructure</span>
		<span class="tag">Land Use</span>
		<span class="tag">Preparedness</span>
		""",
		unsafe_allow_html=True,
	)
	st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='section-heading'>Feature reference</div>", unsafe_allow_html=True)
reference_columns = st.columns(3)
for idx, feature in enumerate(FEATURES):
	with reference_columns[idx % 3]:
		st.markdown(f"- {feature}")
st.markdown("</div>", unsafe_allow_html=True)
