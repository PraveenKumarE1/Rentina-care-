# API contract

- `POST /grade`: multipart image upload. Returns `screening_id`, `grade`, `confidence`, `referable`, and `report_url`.
- `GET /report/{id}`: returns HTML or PDF report.
- `GET /patient/{id}/history`: returns chronological screening records.

Require an API key, TLS termination, request-size limits, audit logging, and de-identification before exposing a gateway. See `frontend/api_server.m`.
