CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS screenings (screening_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, image_id TEXT, screened_at TEXT NOT NULL, quality_passed INTEGER, grade INTEGER, confidence REAL, referable INTEGER, report_path TEXT, FOREIGN KEY(patient_id) REFERENCES patients(patient_id));
CREATE TABLE IF NOT EXISTS corrections (correction_id TEXT PRIMARY KEY, screening_id TEXT NOT NULL, corrected_grade INTEGER NOT NULL, reviewer TEXT, created_at TEXT NOT NULL);
