"""
角膜地形图智能诊断系统 - 风险评估报告生成模块
生成结构化临床风险评估报告，包含手术风险、进展风险、临床建议等
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskAssessmentReport:
    """风险评估报告生成器"""

    SEVERITY_MAP = {
        'Normal': {'level': '正常', 'stars': 0, 'color': '#588157'},
        'Mild KC': {'level': '轻度', 'stars': 1, 'color': '#DDA15E'},
        'Moderate KC': {'level': '中度', 'stars': 2, 'color': '#E07A3A'},
        'Severe KC': {'level': '重度', 'stars': 3, 'color': '#BC4749'},
    }

    SURGERY_RISK = {
        'Normal': {
            'level': '低风险',
            'description': '角膜形态正常，符合激光手术条件',
            'recommendation': '可安全进行LASIK/SMILE激光手术',
        },
        'Mild KC': {
            'level': '中等风险',
            'description': '存在轻度角膜异常，需综合评估',
            'recommendation': '建议完善角膜生物力学检查，谨慎评估手术适应性',
        },
        'Moderate KC': {
            'level': '高风险',
            'description': '角膜形态明显异常，不适合激光手术',
            'recommendation': '不建议进行角膜激光手术，建议角膜交联术治疗',
        },
        'Severe KC': {
            'level': '极高风险',
            'description': '角膜严重变形变薄，绝对禁忌激光手术',
            'recommendation': '绝对禁忌激光手术，需进行角膜移植评估',
        },
    }

    PROGRESSION_RISK = {
        'Normal': {'level': '极低', 'percentage': '< 5%', 'description': '角膜形态稳定，无明显进展风险'},
        'Mild KC': {'level': '中等', 'percentage': '30-50%', 'description': '有进展可能，建议定期随访观察'},
        'Moderate KC': {'level': '较高', 'percentage': '60-80%', 'description': '进展风险较高，需积极干预'},
        'Severe KC': {'level': '高', 'percentage': '> 80%', 'description': '疾病处于活动期，进展风险高'},
    }

    FOLLOW_UP_INTERVAL = {
        'Normal': {'interval': '12个月', 'description': '年度常规体检即可'},
        'Mild KC': {'interval': '3-6个月', 'description': '密切随访，监测进展'},
        'Moderate KC': {'interval': '1-3个月', 'description': '短期随访，评估治疗效果'},
        'Severe KC': {'interval': '1个月', 'description': '高频随访，考虑手术干预'},
    }

    CLINICAL_RECOMMENDATIONS = {
        'Normal': [
            {'text': '可进行近视激光手术术前检查', 'type': 'recommend', 'star_rating': 4},
            {'text': '建议完善角膜厚度和地形图分析', 'type': 'recommend', 'star_rating': 3},
            {'text': '术后定期复查角膜形态', 'type': 'followup', 'star_rating': 0},
            {'text': '注意用眼卫生，避免揉眼', 'type': 'notice', 'star_rating': 0},
        ],
        'Mild KC': [
            {'text': '建议行角膜生物力学检查（Corvis ST）', 'type': 'recommend', 'star_rating': 4},
            {'text': '完善Pentacam眼前节全面分析', 'type': 'recommend', 'star_rating': 3},
            {'text': '考虑角膜交联术（CXL）控制疾病进展', 'type': 'recommend', 'star_rating': 4},
            {'text': '每3-6个月复查角膜地形图', 'type': 'followup', 'star_rating': 0},
            {'text': '避免揉眼，注意用眼卫生', 'type': 'notice', 'star_rating': 0},
        ],
        'Moderate KC': [
            {'text': '近视激光手术 - 禁忌', 'type': 'contraindication', 'star_rating': 0},
            {'text': '建议行角膜交联术（CXL）控制进展', 'type': 'recommend', 'star_rating': 4},
            {'text': '必要时佩戴RGP角膜接触镜矫正视力', 'type': 'recommend', 'star_rating': 3},
            {'text': '加强随访频率，每1-3个月复查', 'type': 'followup', 'star_rating': 0},
            {'text': '避免剧烈运动和眼部碰撞', 'type': 'notice', 'star_rating': 0},
        ],
        'Severe KC': [
            {'text': '近视激光手术 - 绝对禁忌', 'type': 'contraindication', 'star_rating': 0},
            {'text': '建议角膜交联术控制病情', 'type': 'recommend', 'star_rating': 4},
            {'text': '考虑深板层角膜移植或穿透性角膜移植', 'type': 'recommend', 'star_rating': 4},
            {'text': '佩戴RGP或巩膜镜改善视力', 'type': 'recommend', 'star_rating': 3},
            {'text': '加强随访频率，每月复查', 'type': 'followup', 'star_rating': 0},
            {'text': '必要时转诊角膜病专科', 'type': 'recommend', 'star_rating': 3},
        ],
    }

    def generate(
        self,
        prediction_result: Dict,
        explainability_report: Optional[Dict] = None,
        patient_info: Optional[Dict] = None,
    ) -> Dict:
        """
        生成完整的风险评估报告

        Args:
            prediction_result: 模型预测结果 {
                'prediction': str, 'class_name': str,
                'confidence': float, 'probabilities': dict
            }
            explainability_report: 可解释性分析报告
            patient_info: 患者信息 (可选)

        Returns:
            完整的结构化报告
        """
        prediction = prediction_result.get('prediction', 'Normal')
        confidence = prediction_result.get('confidence', 0.0)
        probabilities = prediction_result.get('probabilities', {})

        severity = self.SEVERITY_MAP.get(prediction, self.SEVERITY_MAP['Moderate KC'])
        surgery = self.SURGERY_RISK.get(prediction, self.SURGERY_RISK['Moderate KC'])
        progression = self.PROGRESSION_RISK.get(prediction, self.PROGRESSION_RISK['Moderate KC'])
        follow_up = self.FOLLOW_UP_INTERVAL.get(prediction, self.FOLLOW_UP_INTERVAL['Moderate KC'])
        recommendations = self.CLINICAL_RECOMMENDATIONS.get(prediction, self.CLINICAL_RECOMMENDATIONS['Moderate KC'])
        # 兼容旧格式（字符串列表）和新格式（字典列表）
        recs_list = []
        for rec in recommendations:
            if isinstance(rec, dict):
                recs_list.append(rec)
            else:
                recs_list.append({'text': rec, 'type': 'recommend', 'star_rating': 0})

        report = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_info': patient_info or self._default_patient_info(),
            'diagnosis': {
                'prediction': prediction,
                'class_name': prediction_result.get('class_name', prediction),
                'confidence': f"{confidence:.1%}",
                'severity': severity['level'],
                'severity_stars': severity['stars'],
                'severity_color': severity['color'],
                'probabilities': probabilities,
            },
            'risk_assessment': {
                'surgery_risk': {
                    'level': surgery['level'],
                    'description': surgery['description'],
                    'recommendation': surgery['recommendation'],
                },
                'progression_risk': {
                    'level': progression['level'],
                    'description': progression['description'],
                },
                'follow_up': {
                    'interval': follow_up['interval'],
                    'description': follow_up['description'],
                },
            },
            'clinical_recommendations': recs_list,
        }

        # 添加可解释性分析结果
        if explainability_report:
            report['explainability'] = {
                'regions': explainability_report.get('regions', []),
                'indicators': explainability_report.get('indicators', []),
                'decision_path': explainability_report.get('decision_path', {}),
                'feature_importance': explainability_report.get('feature_importance', {}),
                'abnormal_indicator_count': explainability_report.get('abnormal_indicator_count', 0),
                'total_indicators': explainability_report.get('total_indicators', 0),
            }

        return report

    def _default_patient_info(self) -> Dict:
        return {
            'name': '--',
            'age': '--',
            'gender': '--',
            'exam_device': 'Pentacam HR',
        }

    def generate_text_report(self, report: Dict) -> str:
        """生成纯文本格式的报告"""
        lines = []
        lines.append("=" * 56)
        lines.append("        角膜地形图AI辅助诊断报告")
        lines.append("=" * 56)

        # 患者信息
        pi = report['patient_info']
        lines.append("")
        lines.append("【患者信息】")
        name = pi.get('name', '--')
        age = pi.get('age', '--')
        gender = pi.get('gender', '--')
        lines.append(f"  姓名: {name}    年龄: {age}    性别: {gender}")
        lines.append(f"  检查日期: {report['report_time']}")
        lines.append(f"  检查设备: {pi.get('exam_device', 'Pentacam HR')}")

        # 诊断结果
        diag = report['diagnosis']
        lines.append("")
        lines.append("【诊断结果】")
        lines.append(f"  分类: {diag['class_name']}")
        lines.append(f"  置信度: {diag['confidence']}")
        stars_str = "★" * diag['severity_stars'] + "☆" * (3 - diag['severity_stars'])
        lines.append(f"  严重程度: {stars_str} ({diag['severity']})")

        # 风险评估
        risk = report['risk_assessment']
        lines.append("")
        lines.append("【风险评估】")

        sr = risk['surgery_risk']
        lines.append(f"  ├─ 手术风险等级: {sr['level']}")
        lines.append(f"  │   └─ {sr['description']}")

        pr = risk['progression_risk']
        lines.append(f"  ├─ 疾病进展风险: {pr['level']}")
        lines.append(f"  │   └─ {pr['description']}")

        fu = risk['follow_up']
        lines.append(f"  └─ 建议随访间隔: {fu['interval']}")
        lines.append(f"      └─ {fu['description']}")

        # 可解释性
        if 'explainability' in report:
            exp = report['explainability']
            indicators = exp.get('indicators', [])

            if indicators:
                lines.append("")
                lines.append("【临床指标分析】")
                for ind in indicators:
                    status_mark = "✅" if not ind['abnormal'] else "❌"
                    lines.append(f"  {status_mark} {ind['name']}: {ind['value']} {ind['unit']} "
                                 f"(正常: {ind['normal_range']}) - {ind['status']}")

                abnormal_count = exp.get('abnormal_indicator_count', 0)
                total = exp.get('total_indicators', 0)
                lines.append(f"  异常指标: {abnormal_count}/{total}项")

            regions = exp.get('regions', [])
            if regions:
                lines.append("")
                lines.append("【关键区域分析】")
                for r in regions[:3]:
                    lines.append(f"  • {r['region_type']} - 关注度: {r['avg_attention']:.2f} "
                                 f"面积占比: {r['area_ratio']:.1%} 严重度: {r['severity']}")

        # 临床建议
        lines.append("")
        lines.append("【临床建议】")
        for i, rec in enumerate(report['clinical_recommendations'], 1):
            text = rec['text'] if isinstance(rec, dict) else rec
            rec_type = rec.get('type', 'recommend') if isinstance(rec, dict) else 'recommend'
            if rec_type == 'contraindication':
                lines.append(f"  ✗ {text}")
            elif rec_type == 'recommend':
                star = rec.get('star_rating', 0) if isinstance(rec, dict) else 0
                star_str = f" - 推荐指数{'★' * star}{'☆' * (4 - star)}" if star > 0 else ""
                lines.append(f"  ✓ {text}{star_str}")
            elif rec_type == 'followup':
                lines.append(f"  ◎ {text}")
            else:
                lines.append(f"  • {text}")

        # 免责声明
        lines.append("")
        lines.append("【免责声明】")
        lines.append("  本报告仅供参考，最终诊断请以临床医生判断为准。")

        lines.append("")
        lines.append("=" * 56)
        lines.append(f"        报告生成时间: {report['report_time']}")
        lines.append("=" * 56)

        return "\n".join(lines)
