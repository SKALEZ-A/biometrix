import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import logging
from jinja2 import Template
import os

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.templates_dir = "templates/email"
        self._load_templates()
    
    def _load_templates(self):
        """Load email templates from files."""
        self.templates = {}
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Fraud alert template
        fraud_template_content = """
        <h2>Fraud Alert - {{ system_name }}</h2>
        <p>Dear {{ admin_name }},</p>
        <p>A potential fraud attempt was detected:</p>
        <ul>
            <li><strong>User ID:</strong> {{ user_id }}</li>
            <li><strong>Score:</strong> {{ fraud_score }}</li>
            <li><strong>Severity:</strong> {{ severity }}</li>
            <li><strong>Time:</strong> {{ timestamp }}</li>
            <li><strong>Reason:</strong> {{ reason }}</li>
        </ul>
        <p>Please review immediately: <a href="{{ dashboard_url }}">Dashboard</a></p>
        <p>Best,<br>{{ system_name }} Team</p>
        """
        with open(os.path.join(self.templates_dir, "fraud_alert.html"), "w") as f:
            f.write(fraud_template_content)
        self.templates["fraud_alert"] = Template(fraud_template_content)
        
        # Enrollment confirmation
        enroll_template = """
        <h2>Biometric Enrollment Successful</h2>
        <p>Hello {{ user_name }},</p>
        <p>Your biometric data has been successfully enrolled in {{ system_name }}.</p>
        <p>Enrollment Time: {{ timestamp }}</p>
        <p>You can now use face/fingerprint authentication for secure access.</p>
        <p>If this wasn't you, contact support immediately.</p>
        <p>Regards,<br>{{ system_name }}</p>
        """
        with open(os.path.join(self.templates_dir, "enrollment.html"), "w") as f:
            f.write(enroll_template)
        self.templates["enrollment"] = Template(enroll_template)
    
    def _create_message(self, to_email: str, subject: str, html_body: str) -> MimeMultipart:
        """Create MIME message."""
        msg = MimeMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = to_email
        
        html_part = MimeText(html_body, "html")
        msg.attach(html_part)
        return msg
    
    async def send_async(self, to_email: str, template_name: str, context: Dict[str, Any], subject: str):
        """Async email sending using asyncio."""
        loop = asyncio.get_event_loop()
        
        def _send():
            try:
                html_body = self.templates[template_name].render(**context)
                msg = self._create_message(to_email, subject, html_body)
                
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.username, self.password)
                text = msg.as_string()
                server.sendmail(self.from_email, to_email, text)
                server.quit()
                
                logger.info(f"Email sent to {to_email}: {subject}")
                return True
            except Exception as e:
                logger.error(f"Email send failed to {to_email}: {e}")
                return False
        
        return await loop.run_in_executor(None, _send)
    
    def send_fraud_alert(self, to_email: str, user_id: str, fraud_score: float, severity: str, reason: str, dashboard_url: str = "http://localhost:3000"):
        """Send fraud alert email."""
        context = {
            "system_name": "Biometric Fraud Prevention",
            "admin_name": "Admin",
            "user_id": user_id,
            "fraud_score": f"{fraud_score:.3f}",
            "severity": severity.upper(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "dashboard_url": dashboard_url
        }
        subject = f"🚨 Fraud Alert: {severity.upper()} - User {user_id}"
        
        # For sync use; wrap in async for production
        asyncio.run(self.send_async(to_email, "fraud_alert", context, subject))
    
    def send_enrollment_confirmation(self, to_email: str, user_name: str):
        """Send enrollment confirmation."""
        context = {
            "user_name": user_name,
            "system_name": "Biometric Fraud Prevention",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        subject = "✅ Biometric Enrollment Confirmed"
        
        asyncio.run(self.send_async(to_email, "enrollment", context, subject))

# Global instance (configure from settings)
from config.app_config import settings
email_service = EmailService(
    smtp_server=settings.smtp_server,
    smtp_port=settings.smtp_port,
    username=os.getenv("SMTP_USERNAME", "noreply@biometric-system.com"),
    password=os.getenv("SMTP_PASSWORD", ""),
    from_email=settings.email_from
)

# Batch email sender for alerts
async def send_batch_alerts(alerts: List[Dict], admin_emails: List[str]):
    """Send batch fraud alerts to multiple admins."""
    for alert in alerts:
        for email in admin_emails:
            await email_service.send_async(
                email,
                "fraud_alert",
                {
                    "system_name": "Biometric System",
                    "admin_name": "Admin Team",
                    "user_id": alert["user_id"],
                    "fraud_score": alert["score"],
                    "severity": alert["severity"],
                    "timestamp": alert["created_at"],
                    "reason": alert["reason"],
                    "dashboard_url": "http://localhost:3000/alerts"
                },
                f"🚨 Batch Fraud Alert: {alert['severity']}"
            )
    logger.info(f"Batch alerts sent for {len(alerts)} incidents")

# Integration with logger
def on_fraud_detected(user_id: str, score: float, severity: str, reason: str):
    """Hook for sending email on fraud detection."""
    admin_emails = ["admin@company.com", "security@company.com"]  # From config
    email_service.send_fraud_alert(
        ",".join(admin_emails), user_id, score, severity, reason
    )

if __name__ == "__main__":
    # Test sending
    email_service.send_enrollment_confirmation("test@example.com", "Test User")
    print("Email service test complete.")
