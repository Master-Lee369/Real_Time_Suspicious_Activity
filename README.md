# RealTimeSecurity

Local development notes

Prerequisites:
- Python 3.11+
- pip and virtualenv

Quick start (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # if you have one
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Notes:
- The project uses SQLite for development (`db.sqlite3`). For production, use PostgreSQL and update `RealTimeSecurity/settings.py`.
- Static files are in `static/`. Media uploads use `media/`.
- Basic Bootstrap-based frontend is in `templates/`.

Next recommended steps:
- Add tests for critical behavior.
- Replace SECRET_KEY in production and set `DEBUG = False`.
- Configure a production DB (Postgres) and environment-based settings.
