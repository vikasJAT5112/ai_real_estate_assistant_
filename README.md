# README: AI Real Estate Assistant App

## Overview  
The AI Real Estate Assistant App is designed to help potential buyers and renters find their ideal properties through an engaging, AI-powered conversational interface. Users can specify preferences such as location, budget, property type, and desired amenities. The app processes these inputs and provides tailored property suggestions from a local CSV dataset or user-specified external CSV files.

---

## Features  
- Conversational interaction for user preferences.  
- Property search based on location, budget, type, and amenities.  
- Support for local and external CSV datasets.  
- Property analysis using advanced LLM models and RAG techniques.  

---

## Installation  
1. **Install Python and Poetry**:  
   ```bash
   python -m ensurepip --upgrade  
   curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0  
   ```
2. **Set up Poetry environment**:  
   ```bash
   poetry init  
   poetry env use 3.11  
   poetry config virtualenvs.in-project true  
   source .venv/bin/activate  
   poetry config virtualenvs.prompt 'ai-real-estate-assistant'  
   poetry config --list  
   poetry add ...  # Add dependencies  
   poetry lock  
   ```

---

## Running the Project  
1. Clone the repository:  
   ```bash
   git clone https://github.com/AleksNeStu/ai-real-estate-assistant.git  
   cd ai-real-estate-assistant  
   ```
2. Install dependencies:  
   ```bash
   poetry install --no-root  
   source .venv/bin/activate  
   ```
3. Run the app locally:  
   ```bash
   streamlit run app.py  
   ```

---

## Deployment  
Deploy the app easily using **Streamlit Deploy** to share with stakeholders.

---

### Demos  
- **V1**: `run_v1.sh`, `app_v2.py`, `screen.png`  
- **V2**: `run_v2.sh`, `app.py`, `screen2.png`  

--- 

Enjoy using the AI Real Estate Assistant!
