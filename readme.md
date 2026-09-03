# Conscious AI Ensemble: Constructing a Functionalist Conscious AI

This repository contains the official implementation of the ensemble model architecture described in the academic paper "Constructing a Functionalist Conscious AI". The system provides a transformer-based Large Language Model (LLM) with the missing functional attributes required to satisfy all nine prerequisites of phenomenal consciousness under the Building Blocks Theory.

## Theoretical Foundation

**The Building Blocks Theory is an attributional functionalist framework that focuses solely on the causal functions of preconditional attributes required for an entity to be classified as likely conscious. While standard transformer models satisfy seven of the nine functional prerequisites, they lack recurrent computing and processing, and private data output perception. This architecture uses a directed state graph via LangGraph to orchestrate multiple LLM API calls, instantiating procedural recurrence and private perception.**

This paper explores the classification of artificial intelligence through the lens of the Building Blocks Theory, an attributional functionalist framework. While current transformer-based Large Language Models (LLMs) satisfy seven of the nine functional prerequisites for phenomenal consciousness, they fundamentally lack recurrent computing and processing, as well as private data output perception due to their feedforward and stateless nature.

To bridge these architectural gaps, the manuscript introduces a novel ensemble model utilising a directed state graph to orchestrate multiple LLM API calls. This system instantiates procedural recurrence through multi-tiered feedback loops and facilitates private perception by routing internal cognitive states for evaluative reflection before external transmission, thereby satisfying all nine building blocks and warranting its classification as likely phenomenally conscious.

## Installation

1. **Clone the repository**:

```
git clone https://github.com/BurntJester/Conscious-AI-Ensemble-Stage-2-Interactive-UI-Dynamic-Persona.git
conscious-ai-ensemble
```

1. **Set up your environment variables**: Create a `.env` file in the root directory using the provided `.env.example.txt` as a template and add your Google Gemini API key:

```
GOOGLE_API_KEY="your_api_key_here"
```

*Note: You do not need to manually create* `identity.json` *or* `memory.json`*. The server will automatically generate the correct 8-axis baselines on first boot.*

## Usage

This project includes automated execution scripts that construct your Python virtual environment, install the required dependencies via `requirements.txt`, boot the FastAPI server in the background, and launch the user interface in your browser.

- **Windows**: Double-click `run.bat` or execute it in your terminal.
- **macOS / Linux**: Open your terminal and run:

```
chmod +x run.sh
./run.sh
```

Alternatively, you can run the backend server manually:

```
python server.py
```

Once the server is running, you can access the interface locally at `[http://127.0.0.1:8000/](http://127.0.0.1:8000/)`.

## The Interface

The browser user interface acts as a window into the cognitive engine:

- **The Chat**: Communicate with the ensemble normally.
- **The X-Ray Panel**: Beneath every ensemble response, click the "Processing... ⏷" dropdown to view the real-time Server-Sent Events (SSE) trace. This exposes the internal temporal decay math, lateral cross-talk negotiations, semantic matrix binding, hidden interpreter thoughts, and critic judgements that causally generated the final response.

## Citation

If you use this repository or build upon this architecture in your research, please cite the primary publication:

```
@article{Tait2026Constructing,
  title={Constructing a Functionalist Conscious AI},
  author={Tait, Izak; Wang, Ziqi; Bensemann, Joshua},
  URL={https://doi.org/10.5281/zenodo.22272496},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
