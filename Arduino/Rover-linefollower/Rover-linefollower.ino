// Motor control pins
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 10;
const int ENA = 5;
const int ENB = 6;

// Line sensors: Left (A0), Center (A2), Right (A4)
const int sensorPins[2] = {3, 4};
int sensors[2];

// Ultrasonic sensor pins
const int trigPin = 11;
const int echoPin = 12;
long duration;
int distance;
bool lastLoadState = false;  
unsigned long lastStateChangeTime = 0;
bool stableLoadState = false;


bool returning = false;
bool lineFollowing = false;

const int BUZZER_PIN = 13;

int motorSpeed = 155;
int motorturnSpeed = 190;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  for (int i = 0; i < 2; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  Serial.begin(9600);
}

void loop() {
  // Serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      lineFollowing = true;
      returning = false;
      beepBuzzer();
      Serial.println("STARTED");
    }
    if (cmd == 'F') {
      lineFollowing = false;
      returning = false;
      stopMotors();
      Serial.println("STOPPED");
    }
    if (cmd == 'R') {
      returning = true;
      lineFollowing = false;
      beepBuzzer();
      Serial.println("RETURN MODE ACTIVATED");
    }
  }

// Ultrasonic
distance = getDistance();
bool currentLoadState = (distance > 0 && distance < 20);  // true = loaded

if (currentLoadState != lastLoadState) {
  lastLoadState = currentLoadState;
  if (currentLoadState) {
    Serial.println("ROVER LOADED");
  } else {
    Serial.println("ROVER EMPTY");
  }
}

 checkLoadStatus(); 

  // Line following
if (lineFollowing) {
  sensors[0] = digitalRead(sensorPins[0]);  // Left sensor
  sensors[1] = digitalRead(sensorPins[1]);  // Right sensor

  Serial.print("Sensors: ");
  Serial.print(sensors[0]);
  Serial.print(" ");
  Serial.println(sensors[1]);

  if (sensors[0] == 0 && sensors[1] == 0) {
    // Both on black line → go forward
    Serial.println("→ Forward");
    moveForward();
  } else if (sensors[0] == 0 && sensors[1] == 1) {
    // Left sees black, right sees white → adjust right
    Serial.println("↪ Adjust Right");
    turnRight();
  } else if (sensors[0] == 1 && sensors[1] == 0) {
    // Right sees black, left sees white → adjust left
    Serial.println("↩ Adjust Left");
    turnLeft();
  } else {
    // Both off the line → stop or search
    Serial.println("■ Stop");
    stopMotors();
  }
  
  delay(50);
  beepBuzzer();
  delay(1000);
}

  // Reverse line following
  if (returning) {
    for (int i = 0; i < 3; i++) {
      sensors[i] = digitalRead(sensorPins[i]);
    }
    followLineReverse();
    delay(50);
  }
}

// Ultrasonic distance
int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 30000);
  int dist = duration * 0.034 / 2;
  return dist;
}

// Motor control
void moveForward() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN4, HIGH); digitalWrite(IN3, LOW);
}

void moveBackward() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void turnLeft() {
  analogWrite(ENA, motorturnSpeed);
  analogWrite(ENB, motorturnSpeed);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN4, HIGH); digitalWrite(IN3, LOW);
}

void turnRight() {
  analogWrite(ENA, motorturnSpeed);
  analogWrite(ENB, motorturnSpeed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN4, LOW); digitalWrite(IN3, HIGH);
}

void turnAround() {
  analogWrite(ENA, motorturnSpeed);
  analogWrite(ENB, motorturnSpeed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void beepBuzzer() {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(300);
  digitalWrite(BUZZER_PIN, LOW);
}

void followLineReverse() {
  if (sensors[0] == 0 && sensors[1] == 0 && sensors[2] == 0) {
    stopMotors();
    beepBuzzer();
    Serial.println("● RETURNED TO STATION (STOP LINE)");
    lineFollowing = false;
    returning = false;
    return;
  }

  if (sensors[1] == 0) {
    moveBackward();
    Serial.println("← BACKWARD");
  } else if (sensors[0] == 0) {
    turnRight();  // Reverse logic
    Serial.println("↩ REVERSE RIGHT");
  } else if (sensors[2] == 0) {
    turnLeft();   // Reverse logic
    Serial.println("↪ REVERSE LEFT");
  } else {
    stopMotors();
    Serial.println("■ STOP");
  }
}

void checkLoadStatus() {
  static bool lastReading = false;
  static unsigned long lastDebounceTime = 0;
  const unsigned long debounceDelay = 300;  // 300 ms debounce time

  distance = getDistance();
  bool currentReading = (distance > 0 && distance < 15);  // true = loaded

  if (currentReading != lastReading) {
    lastDebounceTime = millis();  // Reset the debounce timer
    lastReading = currentReading;
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (currentReading != stableLoadState) {
      stableLoadState = currentReading;
      if (stableLoadState) {
        Serial.println("ROVER LOADED");
      } else {
        Serial.println("ROVER EMPTY");
      }
    }
  }
}

