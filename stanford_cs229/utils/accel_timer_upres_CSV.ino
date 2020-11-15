
// constants describing pin allocations
const int xpin = A1;                  // x-axis of the accelerometer
const int ypin = A0;                  // y-axis
//const int zpin = A2;                  // z-axis (only on 3-axis models)

const int pin0 = A2;
const int pin1 = A3;
const int pin2 = A4;
const int pin3 = A5;

void setup() {
  Serial.begin(9600); //initialize serial communications
  pinMode(xpin, INPUT);
  pinMode(ypin, INPUT);
  //pinMode(zpin, INPUT);
  pinMode(pin0, INPUT);
  pinMode(pin1, INPUT);
  pinMode(pin2, INPUT);
  pinMode(pin3, INPUT);
}

//global variables
int dataX; int dataY; int dataZ; //store data from all three axes
int data1; //int data2; int data2; //store data from three muscular sensors
int period = 16; //sampling period (in milliseconds)
int timer = 0; //time elapsed since last sample taken

//output, input tags
String outputX = "X axis acceleration"; String outputY = "Y axis acceleration"; //String outputZ = "Z axis acceleration";
String input1 = "Bicep signal"; String input2 = "Shoulder signal"; String input3 = "Tricep signal"; String input0 = "Forearm signal";
bool label = true;

void loop() {
  //Print out column headers
  while(label){
    Serial.print(outputX);
    Serial.print(",");
    Serial.print(outputY);
    Serial.print(",");
    Serial.print(input1);
    Serial.print(",");
    Serial.print(input2);
    Serial.print(",");
    Serial.print(input3);
    Serial.print(",");
    Serial.print(input0);
    Serial.println(); //new line
    label = false; //only runs once
    timer = millis(); //start the timer
  }

  if (millis() - timer >= period){ //Conditional loop to enforce sampling frequency
    //Display data to Serial Monitor
    Serial.print(analogRead(xpin)); //print sensor value
    //Serial.print("\t"); //print tab between values
    Serial.print(","); //print comma between values
    Serial.print(analogRead(ypin));
    //Serial.print("\t");
    Serial.print(",");
    Serial.print(analogRead(pin1));
    Serial.print(",");
    Serial.print(analogRead(pin2));
    Serial.print(",");
    Serial.print(analogRead(pin3));
    Serial.print(",");
    Serial.print(analogRead(pin0));
    Serial.println(); //new line
    timer = millis(); //reset timer, to delay before taking next reading
  }
}
