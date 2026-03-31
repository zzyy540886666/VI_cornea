"""
角膜地形图智能诊断系统 - 病例管理模块
基于 SQLite 的患者/检查记录/随访管理系统
"""

import sqlite3
import uuid
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _get_default_db_path() -> str:
    """获取数据库路径，兼容 Streamlit Cloud 只读文件系统"""
    # Streamlit Cloud 可写目录: /mount/data 或 /tmp
    candidates = [
        "/mount/data/cases.db",
        os.path.join(os.environ.get("HOME", ""), "cases.db"),
        "/tmp/cases.db",
        "cases.db",  # 本地开发回退
    ]
    for p in candidates:
        try:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            # 测试是否可写
            with open(p, 'a') as f:
                pass
            return p
        except (PermissionError, OSError):
            continue
    return "cases.db"


class CaseManager:
    """病例管理系统"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _get_default_db_path()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """创建数据库表"""
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS examinations (
                exam_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                image_filename TEXT,
                prediction TEXT NOT NULL,
                class_name TEXT,
                confidence REAL,
                risk_level TEXT,
                severity TEXT,
                report_json TEXT,
                exam_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE TABLE IF NOT EXISTS follow_ups (
                follow_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                planned_date DATE NOT NULL,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE INDEX IF NOT EXISTS idx_exams_patient ON examinations(patient_id);
            CREATE INDEX IF NOT EXISTS idx_followups_patient ON follow_ups(patient_id);
            CREATE INDEX IF NOT EXISTS idx_followups_status ON follow_ups(status);
        ''')
        self.conn.commit()

    # ── 患者管理 ──

    def add_patient(self, name: str, age: int = None, gender: str = None,
                    phone: str = None, notes: str = None) -> Dict:
        patient_id = f"P{uuid.uuid4().hex[:8].upper()}"
        self.conn.execute(
            'INSERT INTO patients (patient_id, name, age, gender, phone, notes) VALUES (?, ?, ?, ?, ?, ?)',
            (patient_id, name, age, gender, phone, notes)
        )
        self.conn.commit()
        return self.get_patient(patient_id)

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        row = self.conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
        return dict(row) if row else None

    def search_patients(self, keyword: str = None, limit: int = 50) -> List[Dict]:
        if keyword:
            rows = self.conn.execute(
                '''SELECT * FROM patients
                   WHERE name LIKE ? OR patient_id LIKE ? OR phone LIKE ?
                   ORDER BY updated_at DESC LIMIT ?''',
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT * FROM patients ORDER BY updated_at DESC LIMIT ?', (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_patient(self, patient_id: str, **kwargs) -> bool:
        if not kwargs:
            return False
        kwargs['updated_at'] = datetime.now().isoformat()
        sets = ', '.join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [patient_id]
        self.conn.execute(f'UPDATE patients SET {sets} WHERE patient_id = ?', vals)
        self.conn.commit()
        return True

    def delete_patient(self, patient_id: str) -> bool:
        self.conn.execute('DELETE FROM follow_ups WHERE patient_id = ?', (patient_id,))
        self.conn.execute('DELETE FROM examinations WHERE patient_id = ?', (patient_id,))
        self.conn.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
        self.conn.commit()
        return True

    # ── 检查记录管理 ──

    def add_examination(self, patient_id: str, prediction: str, class_name: str,
                        confidence: float, risk_level: str, severity: str,
                        image_filename: str = None, report_json: Dict = None) -> Dict:
        exam_id = f"E{uuid.uuid4().hex[:8].upper()}"
        report_str = json.dumps(report_json, ensure_ascii=False) if report_json else None
        self.conn.execute(
            '''INSERT INTO examinations
               (exam_id, patient_id, image_filename, prediction, class_name,
                confidence, risk_level, severity, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (exam_id, patient_id, image_filename, prediction, class_name,
             confidence, risk_level, severity, report_str)
        )
        self.conn.commit()

        # 自动创建随访计划
        follow_up_interval = {
            'Normal': 365, 'Mild KC': 90, 'Moderate KC': 30, 'Severe KC': 30
        }
        days = follow_up_interval.get(prediction, 90)
        self.add_follow_up(patient_id, days, f"诊断{class_name}后的常规随访")

        return self.get_examination(exam_id)

    def get_examination(self, exam_id: str) -> Optional[Dict]:
        row = self.conn.execute('SELECT * FROM examinations WHERE exam_id = ?', (exam_id,)).fetchone()
        result = dict(row) if row else None
        if result and result.get('report_json'):
            try:
                result['report'] = json.loads(result['report_json'])
                del result['report_json']
            except json.JSONDecodeError:
                pass
        return result

    def get_patient_examinations(self, patient_id: str, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            'SELECT * FROM examinations WHERE patient_id = ? ORDER BY exam_date DESC LIMIT ?',
            (patient_id, limit)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get('report_json'):
                try:
                    d['report'] = json.loads(d['report_json'])
                    del d['report_json']
                except json.JSONDecodeError:
                    pass
            results.append(d)
        return results

    # ── 随访管理 ──

    def add_follow_up(self, patient_id: str, days_ahead: int = 90, notes: str = None) -> Dict:
        follow_id = f"F{uuid.uuid4().hex[:8].upper()}"
        planned = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        self.conn.execute(
            'INSERT INTO follow_ups (follow_id, patient_id, planned_date, status, notes) VALUES (?, ?, ?, ?, ?)',
            (follow_id, patient_id, planned, 'pending', notes)
        )
        self.conn.commit()
        return self.get_follow_up(follow_id)

    def get_follow_up(self, follow_id: str) -> Optional[Dict]:
        row = self.conn.execute('SELECT * FROM follow_ups WHERE follow_id = ?', (follow_id,)).fetchone()
        return dict(row) if row else None

    def get_pending_follow_ups(self, limit: int = 50) -> List[Dict]:
        today = datetime.now().strftime('%Y-%m-%d')
        rows = self.conn.execute(
            '''SELECT f.*, p.name as patient_name
               FROM follow_ups f JOIN patients p ON f.patient_id = p.patient_id
               WHERE f.status = 'pending' AND f.planned_date <= ?
               ORDER BY f.planned_date ASC LIMIT ?''',
            (today, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_follow_up(self, follow_id: str, notes: str = None) -> bool:
        self.conn.execute(
            'UPDATE follow_ups SET status = ?, notes = COALESCE(?, notes) WHERE follow_id = ?',
            ('completed', notes, follow_id)
        )
        self.conn.commit()
        return True

    # ── 统计 ──

    def get_statistics(self) -> Dict:
        total_patients = self.conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        total_exams = self.conn.execute('SELECT COUNT(*) FROM examinations').fetchone()[0]
        pending_followups = self.conn.execute(
            "SELECT COUNT(*) FROM follow_ups WHERE status = 'pending'"
        ).fetchone()[0]

        distribution = {}
        for row in self.conn.execute(
            'SELECT prediction, COUNT(*) as cnt FROM examinations GROUP BY prediction'
        ).fetchall():
            distribution[row['prediction']] = row['cnt']

        return {
            'total_patients': total_patients,
            'total_examinations': total_exams,
            'pending_follow_ups': pending_followups,
            'prediction_distribution': distribution,
        }

    def close(self):
        self.conn.close()
