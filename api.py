"""
2026 年角膜地形图分类模型 - FastAPI REST 接口
提供 RESTful API 供第三方系统调用
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
from PIL import Image
from pathlib import Path
import logging
from datetime import datetime
import json

# 导入模型服务
from model_service import get_model_service, initialize_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="角膜地形图分类 API",
    description="基于深度学习的角膜地形图智能分类系统 - 辅助近视手术评估",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型服务
model_service = None


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global model_service
    try:
        logger.info("正在加载模型服务...")
        model_service = initialize_service()
        logger.info("模型服务加载成功")
    except Exception as e:
        logger.error(f"模型服务加载失败：{str(e)}")
        raise


# 数据模型
class PredictionResponse(BaseModel):
    """单张图片预测响应"""
    success: bool
    prediction: Optional[str] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    suggestion: Optional[str] = None
    error: Optional[str] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    success: bool
    results: List[PredictionResponse]
    total: int
    abnormal_count: int
    normal_count: int
    timestamp: str


class ModelInfo(BaseModel):
    """模型信息响应"""
    success: bool
    model: dict
    device: str
    timestamp: str


class HealthStatus(BaseModel):
    """健康状态响应"""
    success: bool
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# API 接口
@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "角膜地形图分类 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    健康检查接口
    检查服务是否正常运行
    """
    if model_service is None:
        return HealthStatus(
            success=False,
            status="unhealthy",
            model_loaded=False,
            device="N/A",
            timestamp=datetime.now().isoformat()
        )
    
    health = model_service.health_check()
    return HealthStatus(**health)


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_information():
    """
    获取模型信息
    返回模型名称、版本、准确率等详细信息
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务未初始化"
        )
    
    info = model_service.get_model_info()
    return ModelInfo(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(image: UploadFile = File(..., description="角膜地形图图片文件")):
    """
    单张图片预测接口
    
    - **image**: 上传的角膜地形图图片（JPG/PNG 格式）
    
    返回预测结果，包括类别、置信度、建议等
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务未初始化"
        )
    
    # 验证文件类型
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传图片文件"
        )
    
    try:
        # 读取图片
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 执行预测
        result = model_service.predict(image)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"预测失败：{str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败：{str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(images: List[UploadFile] = File(..., description="角膜地形图图片文件列表")):
    """
    批量预测接口
    
    - **images**: 上传的多张角膜地形图图片
    
    返回所有图片的预测结果和统计信息
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务未初始化"
        )
    
    if len(images) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请至少上传一张图片"
        )
    
    if len(images) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="一次最多上传 100 张图片"
        )
    
    try:
        results = []
        for image_file in images:
            # 验证文件类型
            if not image_file.content_type.startswith('image/'):
                continue
            
            # 读取图片
            image_data = await image_file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # 执行预测
            result = model_service.predict(image)
            result['filename'] = image_file.filename
            results.append(result)
        
        # 统计
        abnormal_count = sum(1 for r in results if r.get('prediction') == 'KC')
        normal_count = sum(1 for r in results if r.get('prediction') == 'Normal')
        
        return BatchPredictionResponse(
            success=True,
            results=results,
            total=len(results),
            abnormal_count=abnormal_count,
            normal_count=normal_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"批量预测失败：{str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量预测失败：{str(e)}"
        )


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"未处理的异常：{str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "服务器内部错误",
            "timestamp": datetime.now().isoformat()
        }
    )


# 主函数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
