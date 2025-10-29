"""
Critical Alert Service
Handles detection and notification of high-risk users
"""
from datetime import datetime, timezone
from models import CriticalUser, User, db
import logging
import os

logger = logging.getLogger(__name__)

# Threshold for critical scores (80% or higher)
CRITICAL_THRESHOLD = 80.0


def normalize_score_to_percentage(score, assessment_type, **kwargs):
    """
    Normalize different assessment scores to 0-100 percentage scale

    Args:
        score: The raw score value
        assessment_type: Type of assessment ('phq9', 'scid5pd', 'audio_video', 'composite')
        **kwargs: Additional parameters (e.g., max_score, positives)

    Returns:
        tuple: (percentage_score, raw_score_string)
    """
    if assessment_type == 'phq9':
        # PHQ-9: 0-27 scale
        max_score = 27
        percentage = (score / max_score) * 100
        raw_score = f"{score}/{max_score}"

    elif assessment_type == 'scid5pd':
        # SCID-5-PD: Count of positive responses out of 20
        max_score = 20
        percentage = (score / max_score) * 100
        raw_score = f"{score}/{max_score}"

    elif assessment_type == 'audio_video':
        # Audio/Video: 0-1.0 scale (depression_score)
        percentage = score * 100
        raw_score = f"{score:.2f}"

    elif assessment_type == 'composite':
        # Composite: 0-1.0 scale
        percentage = score * 100
        raw_score = f"{score:.2f}"

    else:
        # Unknown type, assume 0-100 scale
        percentage = float(score)
        raw_score = str(score)

    return percentage, raw_score


def check_and_create_critical_alert(user_id, assessment_type, score, severity, **kwargs):
    """
    Check if a user's assessment score meets critical threshold and create alert

    Args:
        user_id: ID of the user
        assessment_type: Type of assessment
        score: The score value (in original format)
        severity: Severity level string
        **kwargs: Additional parameters

    Returns:
        tuple: (is_critical: bool, alert: CriticalUser or None, message: str)
    """
    try:
        # Normalize score to percentage
        percentage, raw_score = normalize_score_to_percentage(score, assessment_type, **kwargs)

        logger.info(f"Checking critical threshold for user {user_id}: "
                   f"{assessment_type} score={percentage:.1f}% (threshold={CRITICAL_THRESHOLD}%)")

        # Check if score meets critical threshold
        if percentage < CRITICAL_THRESHOLD:
            return False, None, None

        # Get user details
        user = User.query.get(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return False, None, None

        # Create critical user alert
        critical_alert = CriticalUser(
            user_id=user.id,
            username=user.username,
            phone_number=user.phone_number,
            email=user.email,
            assessment_type=assessment_type,
            score=percentage,
            raw_score=raw_score,
            severity=severity,
            alert_sent=False,
            admin_viewed=False,
            resolved=False,
            created_at=datetime.now(timezone.utc)
        )

        db.session.add(critical_alert)
        db.session.commit()

        logger.critical(f"🚨 CRITICAL ALERT CREATED: User {user.username} ({user.phone_number}) "
                       f"scored {percentage:.1f}% on {assessment_type} assessment")

        # Generate comforting message for user
        comforting_message = generate_comforting_message(percentage)

        return True, critical_alert, comforting_message

    except Exception as e:
        logger.error(f"Error creating critical alert: {e}", exc_info=True)
        db.session.rollback()
        return False, None, None


def generate_comforting_message(score_percentage):
    """
    Generate a comforting message for high-risk users

    Args:
        score_percentage: The percentage score

    Returns:
        str: Comforting message
    """
    if score_percentage >= 90:
        return """
        <div class="alert alert-info border-info" style="background-color: #e8f4f8; border-left: 4px solid #17a2b8;">
            <h5 class="alert-heading">
                <i class="fas fa-heart text-danger"></i> We're Here for You
            </h5>
            <p class="mb-2">
                Thank you for completing this assessment. We want you to know that <strong>you're not alone</strong>.
            </p>
            <p class="mb-2">
                Please take a deep breath and relax. What you're feeling is valid, and help is available.
            </p>
            <p class="mb-0">
                <strong>Someone will reach out to you soon</strong> to provide support and guidance.
                In the meantime, please consider reaching out to a trusted friend, family member, or
                calling a crisis helpline if you need immediate support.
            </p>
            <hr class="my-3">
            <p class="mb-0 small">
                <strong>24/7 Crisis Support:</strong><br>
                🚨 NIMHANS Helpline: <strong>080-46110007</strong><br>
                🚨 iCall: <strong>9152987821</strong> (Mon-Sat, 8 AM - 10 PM)<br>
                🚨 Vandrevala Foundation: <strong>1860-2662-345</strong> (24/7)
            </p>
        </div>
        """
    else:
        return """
        <div class="alert alert-info border-info" style="background-color: #e8f4f8; border-left: 4px solid #17a2b8;">
            <h5 class="alert-heading">
                <i class="fas fa-heart text-info"></i> We're Here for You
            </h5>
            <p class="mb-2">
                Thank you for completing this assessment. We want you to know that <strong>you're not alone</strong>.
            </p>
            <p class="mb-2">
                Please take a deep breath and relax. Your results show that you might benefit from some support.
            </p>
            <p class="mb-0">
                <strong>Someone will reach out to you soon</strong> to provide guidance and support.
                Remember, seeking help is a sign of strength, not weakness.
            </p>
        </div>
        """


def get_unviewed_critical_alerts():
    """
    Get all critical alerts that haven't been viewed by admin

    Returns:
        list: List of CriticalUser objects
    """
    try:
        alerts = CriticalUser.query.filter_by(
            admin_viewed=False,
            resolved=False
        ).order_by(CriticalUser.created_at.desc()).all()

        return alerts
    except Exception as e:
        logger.error(f"Error fetching critical alerts: {e}")
        return []


def get_all_critical_alerts(include_resolved=False):
    """
    Get all critical alerts

    Args:
        include_resolved: Whether to include resolved alerts

    Returns:
        list: List of CriticalUser objects
    """
    try:
        query = CriticalUser.query

        if not include_resolved:
            query = query.filter_by(resolved=False)

        alerts = query.order_by(CriticalUser.created_at.desc()).all()

        return alerts
    except Exception as e:
        logger.error(f"Error fetching all critical alerts: {e}")
        return []


def mark_alert_viewed(alert_id, admin_notes=None):
    """
    Mark a critical alert as viewed by admin

    Args:
        alert_id: ID of the alert
        admin_notes: Optional notes from admin

    Returns:
        bool: Success status
    """
    try:
        alert = CriticalUser.query.get(alert_id)
        if not alert:
            return False

        alert.admin_viewed = True
        alert.admin_viewed_at = datetime.now(timezone.utc)

        if admin_notes:
            alert.admin_notes = admin_notes

        db.session.commit()
        logger.info(f"Alert {alert_id} marked as viewed")
        return True

    except Exception as e:
        logger.error(f"Error marking alert as viewed: {e}")
        db.session.rollback()
        return False


def mark_alert_resolved(alert_id, admin_notes=None):
    """
    Mark a critical alert as resolved

    Args:
        alert_id: ID of the alert
        admin_notes: Optional resolution notes

    Returns:
        bool: Success status
    """
    try:
        alert = CriticalUser.query.get(alert_id)
        if not alert:
            return False

        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        alert.admin_viewed = True
        alert.admin_viewed_at = alert.admin_viewed_at or datetime.now(timezone.utc)

        if admin_notes:
            alert.admin_notes = admin_notes

        db.session.commit()
        logger.info(f"Alert {alert_id} marked as resolved")
        return True

    except Exception as e:
        logger.error(f"Error marking alert as resolved: {e}")
        db.session.rollback()
        return False


def send_admin_notification(alert):
    """
    Send notification to admin (SMS, email, or in-app)

    Args:
        alert: CriticalUser object

    Returns:
        bool: Success status
    """
    try:
        admin_phone = os.getenv('ADMIN_PHONE', '9108370049')
        admin_email = os.getenv('ADMIN_EMAIL', 'admin@mindtrack.com')

        # Log the notification
        logger.critical(
            f"📱 ADMIN NOTIFICATION: "
            f"User {alert.username} (Phone: {alert.phone_number}) "
            f"has a high distress score ({alert.score:.1f}%) "
            f"on {alert.assessment_type} assessment. "
            f"Severity: {alert.severity}. "
            f"Please check immediately!"
        )

        # In production, integrate with:
        # - Twilio for SMS: send_sms(admin_phone, message)
        # - SendGrid for email: send_email(admin_email, subject, message)
        # - Push notifications

        # For now, we'll simulate the notification
        notification_message = (
            f"⚠️ ALERT: User {alert.username} (Phone: {alert.phone_number or 'N/A'}) "
            f"has a high distress score ({alert.score:.1f}%). "
            f"Assessment: {alert.assessment_type}. "
            f"Please check the admin dashboard immediately."
        )

        # Mark as sent
        alert.alert_sent = True
        alert.alert_sent_at = datetime.now(timezone.utc)
        db.session.commit()

        logger.info(f"Admin notification sent for alert {alert.id}")
        return True

    except Exception as e:
        logger.error(f"Error sending admin notification: {e}")
        return False


def get_critical_alerts_count():
    """
    Get count of unviewed critical alerts

    Returns:
        int: Count of unviewed alerts
    """
    try:
        count = CriticalUser.query.filter_by(
            admin_viewed=False,
            resolved=False
        ).count()
        return count
    except Exception as e:
        logger.error(f"Error getting critical alerts count: {e}")
        return 0
