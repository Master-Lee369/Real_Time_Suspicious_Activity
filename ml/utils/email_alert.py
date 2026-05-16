"""
Simple email alert helper using SMTP.
Call `send_alert(subject, message, to_email)` when an alert should be sent.
Settings are read from environment variables for safety.
"""
import os
import smtplib
from email.mime.text import MIMEText


def send_alert(subject, message, to_email):
    smtp_host = os.environ.get('ALERT_SMTP_HOST')
    smtp_port = int(os.environ.get('ALERT_SMTP_PORT', 587))
    smtp_user = os.environ.get('ALERT_SMTP_USER')
    smtp_pass = os.environ.get('ALERT_SMTP_PASS')
    from_email = smtp_user
    if not smtp_host or not smtp_user or not smtp_pass:
        print('SMTP not configured; cannot send email alert')
        return False
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    try:
        s = smtplib.SMTP(smtp_host, smtp_port)
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.sendmail(from_email, [to_email], msg.as_string())
        s.quit()
        return True
    except Exception as e:
        print('Failed to send email:', e)
        return False
