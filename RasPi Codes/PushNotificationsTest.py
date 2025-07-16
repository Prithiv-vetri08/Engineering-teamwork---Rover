import unittest
from unittest.mock import patch
import PushNotifications  

class SimpleEmailTest(unittest.TestCase):
    @patch('PushNotifications.smtplib.SMTP_SSL')
    def test_send_email_runs(self, mock_smtp):
        PushNotifications.send_email("Test", "This is a test")
        # If no exceptions, test passes

if __name__ == '__main__':
    unittest.main()
