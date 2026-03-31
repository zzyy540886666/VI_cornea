"""
角膜地形图智能诊断系统 - FastAPI REST 接口 (v2.0 增强版)
支持: 4分类预测 / 可解释性分析 / 风险评估 / 病例管理
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
from PIL import Image
from pathlib import Path
import logging
from datetime import datetime
import json

from model_service import get_model_service, initialize_service
from case_manager import CaseManager
from risk_assessment import RiskAssessmentReport

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="角膜地形图智能诊断 API",
    description="基于深度学习的角膜地形图智能分类系统 v2.0 - 四分类诊断+可解释性+风险评估",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = None
case_manager = None


@app.on_event("startup")
async def startup_event():
    global model_service, case_manager
    try:
        logger.info("正在加载模型服务...")
        model_service = initialize_service()
        logger.info("模型服务加载成功")
    except Exception as e:
        logger.error(f"模型服务加载失败：{str(e)}")
        raise
    try:
        case_manager = CaseManager()
        logger.info("病例管理系统初始化成功")
    except Exception as e:
        logger.error(f"病例管理系统初始化失败：{str(e)}")


# ── 数据模型 ──

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    severity: Optional[int] = None
    suggestion: Optional[str] = None
    explainability: Optional[dict] = None
    risk_report: Optional[dict] = None
    error: Optional[str] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total: int
    abnormal_count: int
    normal_count: int
    timestamp: str


class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class ExaminationCreate(BaseModel):
    patient_id: str
    prediction: str
    class_name: str
    confidence: float
    risk_level: str
    severity: str
    image_filename: Optional[str] = None


# ── API 接口 ──

@app.get("/", tags=["Root"])
async def root():
    return {"message": "角膜地形图智能诊断 API v2.0", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health_check():
    if model_service is None:
        return {"success": False, "status": "unhealthy"}
    return model_service.health_check()


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")
    return model_service.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    image: UploadFile = File(..., description="角膜地形图图片"),
    explain: bool = True,
):
    """
    单张图片预测接口
    - **image**: 上传的角膜地形图图片（JPG/PNG）
    - **explain**: 是否启用可解释性分析（默认True）
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = model_service.predict(pil_image, enable_explainability=explain)

        # 移除 heatmap_overlay_bytes（不适合 JSON 返回）
        response = PredictionResponse(
            success=result.get('success', False),
            prediction=result.get('prediction'),
            class_name=result.get('class_name'),
            confidence=result.get('confidence'),
            probabilities=result.get('probabilities'),
            severity=result.get('severity'),
            suggestion=result.get('suggestion'),
            explainability=result.get('explainability'),
            risk_report=result.get('risk_report'),
            error=result.get('error'),
            timestamp=datetime.now().isoformat()
        )
        return response
    except Exception as e:
        logger.error(f"预测失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


@app.post("/predict/heatmap", tags=["Prediction"])
async def predict_with_heatmap(
    image: UploadFile = File(..., description="角膜地形图图片"),
):
    """预测并返回热力图叠加图片"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")

    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = model_service.predict(pil_image, enable_explainability=True)

        overlay_bytes = result.get('heatmap_overlay_bytes')
        if overlay_bytes:
            return StreamingResponse(
                io.BytesIO(overlay_bytes),
                media_type="image/png",
                headers={"X-Prediction": result.get('class_name', ''),
                         "X-Confidence": f"{result.get('confidence', 0):.2%}"}
            )
        else:
            raise HTTPException(status_code=500, detail="热力图生成失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(images: List[UploadFile] = File(...)):
    """批量预测"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="模型服务未初始化")
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")
    if len(images) > 100:
        raise HTTPException(status_code=400, detail="一次最多上传100张")

    results = []
    for f in images:
        if not f.content_type.startswith('image/'):
            continue
        image_data = await f.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        r = model_service.predict(pil_image, enable_explainability=False)
        r['filename'] = f.filename
        if 'heatmap_overlay_bytes' in r:
            del r['heatmap_overlay_bytes']
        results.append(r)

    return BatchPredictionResponse(
        success=True, results=results, total=len(results),
        abnormal_count=sum(1 for r in results if r.get('prediction') not in ('Normal',)),
        normal_count=sum(1 for r in results if r.get('prediction') == 'Normal'),
        timestamp=datetime.now().isoformat()
    )


# ── 病例管理接口 ──

@app.post("/patients", tags=["Case Management"])
async def create_patient(patient: PatientCreate):
    """新建患者"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    result = case_manager.add_patient(**patient.model_dump())
    return {"success": True, "data": result}


@app.get("/patients", tags=["Case Management"])
async def search_patients(keyword: str = None, limit: int = 50):
    """搜索患者"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    results = case_manager.search_patients(keyword=keyword, limit=limit)
    return {"success": True, "data": results}


@app.get("/patients/{patient_id}", tags=["Case Management"])
async def get_patient(patient_id: str):
    """获取患者详情"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    result = case_manager.get_patient(patient_id)
    if not result:
        raise HTTPException(status_code=404, detail="患者不存在")
    return {"success": True, "data": result}


@app.get("/patients/{patient_id}/examinations", tags=["Case Management"])
async def get_patient_exams(patient_id: str):
    """获取患者检查记录"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    results = case_manager.get_patient_examinations(patient_id)
    return {"success": True, "data": results}


@app.get("/statistics", tags=["Case Management"])
async def get_statistics():
    """获取统计数据"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    stats = case_manager.get_statistics()
    return {"success": True, "data": stats}


@app.get("/follow-ups/pending", tags=["Case Management"])
async def get_pending_followups():
    """获取待随访列表"""
    if case_manager is None:
        raise HTTPException(status_code=503, detail="病例管理系统未初始化")
    results = case_manager.get_pending_follow_ups()
    return {"success": True, "data": results}


# ── 错误处理 ──

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={
        "success": False, "error": exc.detail,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
