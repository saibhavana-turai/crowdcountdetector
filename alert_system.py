# filename: app/alert_system.py

import smtplib
from email.mime.text import MIMEText
import threading

def send_alert_in_background(count, receiver_email):
    """
    This function contains the actual email sending logic and is run in a
    separate thread to avoid blocking the Streamlit app.
    """
    # --- IMPORTANT: CONFIGURE YOUR CREDENTIALS HERE ---
    # You must use a 16-digit "App Password" from your Google Account, not your regular password.
    sender = "youremail@example.com"      # <-- CHANGE THIS to your Gmail address
    password = "your-16-digit-app-password"  # <-- CHANGE THIS to your App Password
    # ----------------------------------------------------

    subject = "CrowdSense Alert: High Crowd Density Detected"
    body = f"Alert: The detected crowd count has reached {count}, exceeding your set threshold."

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receiver_email

        # Connect to Gmail's SMTP server over SSL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print(f"✅ Email alert successfully sent to {receiver_email}.")
    except Exception as e:
        # Print the error to the terminal where Streamlit is running
        print(f"❌ Failed to send email alert: {e}")

def send_alert(count, receiver_email):
    """
    This is the main function called by the Streamlit app.
    It starts the email sending process in a non-blocking background thread.
    """
    # Running in a thread prevents the Streamlit app from freezing while the email is being sent.
    email_thread = threading.Thread(
        target=send_alert_in_background,
        args=(count, receiver_email)
    )
    email_thread.start()