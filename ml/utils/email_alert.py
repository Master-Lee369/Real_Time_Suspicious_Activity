"""Simple, safe email alert helper.

This module reads SMTP/email settings from environment variables:

- EMAIL_HOST
- EMAIL_PORT
- EMAIL_USE_TLS (True/False)
- EMAIL_HOST_USER
- EMAIL_HOST_PASSWORD
- ALERT_RECEIVER_EMAIL

If those are not present, the functions will log a warning and return without raising.
"""
import os
import logging
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _get_smtp_config():
    # Prefer EMAIL_* vars, but keep compatibility with ALERT_SMTP_* if present
    host = os.environ.get('EMAIL_HOST') or os.environ.get('ALERT_SMTP_HOST')
    port = os.environ.get('EMAIL_PORT') or os.environ.get('ALERT_SMTP_PORT') or 587
    use_tls = os.environ.get('EMAIL_USE_TLS')
    if use_tls is None:
        use_tls = os.environ.get('ALERT_SMTP_USE_TLS', 'True')
    use_tls = str(use_tls).lower() in ('1', 'true', 'yes')
    user = os.environ.get('EMAIL_HOST_USER') or os.environ.get('ALERT_SMTP_USER')
    password = os.environ.get('EMAIL_HOST_PASSWORD') or os.environ.get('ALERT_SMTP_PASS')
    return {
        'host': host,
        'port': int(port),
        'use_tls': use_tls,
        'user': user,
        'password': password,
    }


def send_security_alert(subject: str, message: str) -> bool:
    """Send a security alert email to the configured receiver.

    Returns True on success, False otherwise. If email settings or receiver
    are missing, logs a warning and returns False.
    """
    cfg = _get_smtp_config()
    receiver = os.environ.get('ALERT_RECEIVER_EMAIL')

    if not cfg['host'] or not cfg['user'] or not cfg['password']:
        logger.warning('Email settings missing; cannot send security alert')
        return False
    if not receiver:
        logger.warning('ALERT_RECEIVER_EMAIL not configured; cannot send security alert')
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = cfg['user']
    msg['To'] = receiver
    msg.set_content(message)

    try:
        if cfg['use_tls']:
            smtp = smtplib.SMTP(cfg['host'], cfg['port'], timeout=10)
            smtp.starttls()
        else:
            smtp = smtplib.SMTP(cfg['host'], cfg['port'], timeout=10)

        smtp.login(cfg['user'], cfg['password'])
        smtp.send_message(msg)
        smtp.quit()
        logger.info('Security alert sent to %s', receiver)
        return True
    except Exception as e:
        logger.exception('Failed to send security alert: %s', e)
        return False


# Backwards-compatible helper
def send_alert(subject, message, to_email=None):
    """Compatibility wrapper. If `to_email` is provided it sends to that address,
    otherwise it sends to `ALERT_RECEIVER_EMAIL`.
    """
    if to_email:
        # attempt a direct send using provided recipient
        cfg = _get_smtp_config()
        if not cfg['host'] or not cfg['user'] or not cfg['password']:
            logger.warning('Email settings missing; cannot send alert')
            return False
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = cfg['user']
        msg['To'] = to_email
        msg.set_content(message)
        try:
            smtp = smtplib.SMTP(cfg['host'], cfg['port'], timeout=10)
            if cfg['use_tls']:
                smtp.starttls()
            smtp.login(cfg['user'], cfg['password'])
            smtp.send_message(msg)
            smtp.quit()
            return True
        except Exception:
            logger.exception('Failed to send alert')
            return False
    else:
        return send_security_alert(subject, message)
