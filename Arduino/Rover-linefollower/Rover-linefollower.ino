// Motor control pins
const int IN1 = 10;  // Left motor IN1
const int IN2 = 11;  // Left motor IN2
const int IN3 = 12;  // Right motor IN3
const int IN4 = 13;  // Right motor IN4
const int ENA = 5;   // PWM pin for left motor
const int ENB = 6;   // PWM pin for right motor

// Working sensor pins: s1 to s4 and s6 to s8 (excluding s5 on pin 6)
const int sensorPins[7] = {A0, A1, A2, A3, A4, 2, 3};  // s1–s4, s6–s8
int sensors[7];  // Stores readings from 7 working sensors

void setup() {
  // Initialize motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Initialize working sensor pins
  for (int i = 0; i < 7; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  Serial.begin(9600);
}

void loop() {
  // Read 7 sensors
  for (int i = 0; i < 7; i++) {
    sensors[i] = digitalRead(sensorPins[i]);
  }

  // Print sensor values
  for (int i = 0; i < 7; i++) {
    Serial.print(sensors[i]);
    Serial.print(" ");
  }

  // Movement logic based on surrounding sensors
  // Middle: s3 (pin 4) and s4 (pin 5)
  // Left: s1–s2 (pins 2–3)
  // Right: s6–s7 (pins 8–9)

  if (sensors[2] == 0 && sensors[3] == 0) {
    Serial.println(" -> Forward");
    moveForward();
  } else if (sensors[0] == 0 || sensors[1] == 0) {
    Serial.println(" -> Left");
    turnLeft();
  } else if (sensors[4] == 0 || sensors[5] == 0 || sensors[6] == 0) {
    Serial.println(" -> Right");
    turnRight();
  } else {
    Serial.println(" -> Stop");
    stopMotors();
  }

  delay(50);
}

// Motor control functions
void moveForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void turnLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void turnRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
