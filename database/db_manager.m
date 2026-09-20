function varargout=db_manager(action,dbFile,varargin)
%DB_MANAGER Lightweight SQLite persistence for screening and corrections.
if nargin<2 || isempty(dbFile), dbFile='dr_screening.sqlite'; end
if ~ismember(lower(action),{'init','addscreening','history','stats','addcorrection'}), error('db_manager:Action','Unknown action.'); end
conn=sqlite(dbFile,'create'); cleanup=onCleanup(@()close(conn));
switch lower(action)
 case 'init', exec(conn,fileread(fullfile(fileparts(mfilename('fullpath')),'schema.sql'))); varargout={true};
 case 'addscreening'
  r=varargin{1}; exec(conn,'INSERT OR REPLACE INTO screenings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',{char(string(java.util.UUID.randomUUID)),char(r.patientId),char(r.imageId),char(datetime('now')),r.quality.passed,r.grading.level,r.grading.confidence,r.grading.referable,char(r.reportPath)}); varargout={true};
 case 'history', varargout={fetch(conn,'SELECT * FROM screenings WHERE patient_id = ? ORDER BY screened_at',{char(varargin{1})})};
 case 'stats', varargout={fetch(conn,'SELECT grade, COUNT(*) AS total, AVG(referable) AS referral_rate FROM screenings GROUP BY grade')};
 case 'addcorrection', exec(conn,'INSERT INTO corrections VALUES (?, ?, ?, ?, ?)',{char(string(java.util.UUID.randomUUID)),char(varargin{1}),varargin{2},char(varargin{3}),char(datetime('now'))}); varargout={true};
end
end
