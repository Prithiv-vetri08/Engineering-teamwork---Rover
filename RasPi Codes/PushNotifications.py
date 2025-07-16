import smtplib
from email.message import EmailMessage

EMAIL_ADDRESS = 'srhrobocourier@gmail.com'
EMAIL_PASSWORD = 'krkh bhhf fjlc qhsg   '  # Use Gmail App Password
TO_EMAIL = 'mustafadalal748@gmail.com'

def send_email(subject, body):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Example use
send_email("Rover Alert", "Obstacle detected 30 cm ahead!")
