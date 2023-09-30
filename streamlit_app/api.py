from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from typing import Dict
from forecast import Forecast

app = FastAPI()

@app.post("/forecast")
async def instant(data: Dict[str, str]):
    # rjson = await request.json()
    forecast = Forecast(data['stock'], data['date_start'], data['date_end'])
    output = forecast.forecast()
    return JSONResponse(status_code=200, content=output)