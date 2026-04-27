from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Importing the AI function from your ai/inference.py file
from ai.inference import process_password 

app = FastAPI(title="TrustMiiBro Password Enhancer API")

# Allows your frontend HTML to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class PasswordRequest(BaseModel):
    weak_password: str
    target_len: int = 16
    inc_upper: bool = True
    inc_num: bool = True
    inc_sym: bool = True
    inc_ambig: bool = False

class PasswordResponse(BaseModel):
    original_password: str
    enhanced_password: str
    score_percentage: int
    strength_label: str

@app.post("/enhance", response_model=PasswordResponse)
def api_enhance_password(request: PasswordRequest):
    try:
        # Call the AI function with all user preferences
        ai_result = process_password(
            request.weak_password, 
            request.target_len,
            request.inc_upper,
            request.inc_num,
            request.inc_sym,
            request.inc_ambig
        )
        
        # Map the 0-4 AI score to a percentage (20-100%) for the frontend progress bar
        score = ai_result.get("strength_score", 0)
        percentage_map = {0: 20, 1: 40, 2: 60, 3: 80, 4: 100}
        score_percentage = percentage_map.get(score, 20)
        
        return PasswordResponse(
            original_password=request.weak_password,
            enhanced_password=ai_result["enhanced"],
            score_percentage=score_percentage, 
            strength_label=ai_result["strength_label"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)