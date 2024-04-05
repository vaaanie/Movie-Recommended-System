#include <stdio.h>
#include<string.h>
#include<iostream>
#include<conio.h>
#include <windows.h>
using namespace std;
 
 void gotoxy(int x,int y)
{
	COORD CRD;
    CRD.X = x;
    CRD.Y = y;
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE),CRD);
}

// Node Class
class Node {
public:
    int roll;
    string firstName;
    string lastName;
    string Dept;
    float CGPA;
    Node* next;
};
 
// Stores the head of the Linked List
Node* head = new Node();
 
// Check Function to check that if
// Record Already Exist or Not
bool check(int x)
{
    // Base Case
    if (head == NULL)
        return false;
 
    Node* t = new Node;
    t = head;
 
    // Traverse the Linked List
    while (t != NULL) {
        if (t->roll == x)
            return true;
        t = t->next;
    }
 
    return false;
}
// Function to insert the record
void Insert_Record(int roll, string firstName,string lastName,
                   string Dept, float CGPA)
{
    // Create new Node to Insert Record
    Node* t = new Node();
    t->roll = roll;
    t->firstName = firstName;
    t->lastName=lastName;
    t->Dept = Dept;
    t->CGPA = CGPA;
    t->next = NULL;
 
    // Insert at Begin
    if (head == NULL
        || (head->roll >= t->roll)) {
        t->next = head;
        head = t;
    }
 
    // Insert at middle or End
    else {
        Node* c = head;
        while (c->next != NULL
               && c->next->roll < t->roll) {
            c = c->next;
        }
        t->next = c->next;
        c->next = t;
    }
 
    cout << "Record Inserted "
         << "Successfully\n";
}
 
// Function to search record for any
// students Record with roll number

void Search_Record(int roll)
{
    cout<<"Searching for the Student::"<<endl;
    for(int i=0; i<50; i++)
        {
            printf("--");
            Sleep(20);
        }
        cout<<endl;
    // if head is NULL
    if (!head) {
        cout << "No such Record "
             << "Available\n";
        return;
    }
 
    // Otherwise
    else {
        Node* p = head;
        while (p) {
            if (p->roll == roll) {
                cout << "Roll Number\t"
                     << p->roll << endl;
                cout << "Name\t\t"
                     << p->firstName<<" "<<p->lastName << endl;
                cout << "Department\t"
                     << p->Dept << endl;
                cout << "CGPA\t\t"
                     << p->CGPA << endl;
                     system("pause");
                return;
            }
            p = p->next;
        }
 
        if (p == NULL){
            cout << "No such Record "
                 << "Available\n";}
          for(int i=0; i<50; i++)
        {
            printf("--");
            Sleep(20);
        }
        cout<<endl;
    }
    
}
 
 
// Function to delete record students
// record with given roll number
// if it exist
int Delete_Record(int roll)
{
    Node* t = head;
    Node* p = NULL;
 
    // Deletion at Begin
    if (t != NULL
        && t->roll == roll) {
        head = t->next;
        delete t;
 
        cout << "Record Deleted "
             << "Successfully\n";
        return 0;
    }
 
    // Deletion Other than Begin
    while (t != NULL && t->roll != roll) {
        p = t;
        t = t->next;
    }
    if (t == NULL) {
        cout << "Record does not Exist\n";
        return -1;
        p->next = t->next;
 
        delete t;
        cout << "Record Deleted "
             << "Successfully\n";
 
        return 0;
    }
}

// Modify the Record
 void Modify_record(int roll){
if (check(roll))
{
   Node *p =head;
    while (p) {
            if (p->roll == roll) {
                cout << "Roll Number\t"
                     << p->roll << endl;
                cout << "Name\t\t"
                     << p->firstName<<" "<<p->lastName << endl;
                cout << "Department\t"
                     << p->Dept << endl;
                cout << "CGPA\t\t"
                     << p->CGPA << endl;
                cout<<"Select the entry to be modified::"<<endl
                    <<"1.Roll Number\n2.First Name\n3.Last Name\n4.Department\n5.CGPA"<<endl;

                int option;
                cout<<"Enter the choice"<<endl;
                cin>>option;
                if (option==1)
                {
                    cout<<"Enter the new roll no.--"<<endl;
                    int data;
                    cin>>data;
                    p->roll=data;
                }
                else if (option==2)
                {
                    cout<<"Enter the new First Name--"<<endl;
                    string data;
                    cin>>data;
                    p->firstName=data;
                }
                else if (option==3)
                {
                    cout<<"Enter the new Last Name--"<<endl;
                    string data;
                    cin>>data;
                    p->lastName=data;
                }
                else if (option==4)
                {
                    cout<<"Enter the new Department--"<<endl;
                    string data;
                    cin>>data;
                    p->Dept=data;
                }
                else if (option==5)
                {
                    cout<<"Enter the new CGPA --"<<endl;
                    float data;
                    cin>>data;
                    p->CGPA=data;
                }
                 cout << "Record Modified "
         << "Successfully\n";
         return;
            }
            p = p->next;
        }
}
else cout<<"Roll No. Does'nt exist in record"<<endl;
 }
// Function to display the Student's
// Record
void Show_Record()
{
    Node* p = head;
    cout<<"Loading The Data::"<<endl;
    for(int i=0; i<50; i++)
        {
            printf("--");
            Sleep(20);
        }
        cout<<endl;
    if (p == NULL) {
        cout << "No Record "
             << "Available\n";
    }
    else {
        cout << "\tRoll NO.\tName\t\tCourse"
             << "\t\tCGPA\n";
 
        // Until p is not NULL
        while (p != NULL) {
            cout<<"\t" << p->roll << "\t\t"
                 << p->firstName << " "
                 <<p->lastName<<"\t"
                 << p->Dept << "\t"
                 << p->CGPA << endl;
            p = p->next;
        }
    }
    system("pause");
}
 
// Driver code
int main()
{
    head = NULL;
    string firstName,lastName, Dept;
    int Roll;
    float CGPA;
     int Choice;
    // Menu-driven program
    system("color 9f");
    gotoxy(42,8);

		system("cls");
		gotoxy(10,3);
		printf("<<<< Loading Please Wait >>>>");
		for(int i=0; i<5; i++)
        {
            printf("\t(*_*)");
            Sleep(500);
        }
       
 cout<<"\n\n\n\t*******************  Welcome to Student Record Management System  ****************\n";
		printf("\n\n\n\n\n\t\t\t\t\tPress any key to continue......\n ");
		getch();
    do{
        for(int i=0; i<50; i++)
        {
            printf("--");
            Sleep(15);
        }
        cout<<endl;
        cout <<"\n\tSelect :\n\t1."
                "Create a new Record\n\t2.Delete a "
                "student record\n\t3.Search a Student "
                "Record\n\t4.View student record "
                "\n\t5.Modify the Student record"
                "\n\t6.Exit\n";
        cout << "\nEnter your Choice\n";
       
  // Enter Choice
        cin >> Choice;
        if (Choice == 1) {
            cout << "Enter First Name of Student\n";
            cin >> firstName;
            cout << "Enter Last Name of Student\n";
            cin >> lastName;
            cout << "Enter Roll Number of Student\n";
            cin >> Roll;
            if (check(Roll))
            {
                 cout << "Student with this "
             << "record Already Exists\n";
                continue;
            }
            cout << "Enter Course of Student \n";
            cin >> Dept;
            cout << "Enter CGPA of Student\n";
            cin >> CGPA;
            Insert_Record(Roll, firstName,lastName, Dept, CGPA);
        }
        else if (Choice == 2) {
            cout << "Enter Roll Number of Student whose "
                    "record is to be deleted\n";
            cin >> Roll;
            Delete_Record(Roll);
        }
        else if (Choice == 3) {
            cout << "Enter Roll Number of Student whose "
                    "record you want to Search\n";
            cin >> Roll;
            Search_Record(Roll);
        }
        else if (Choice == 4) {
            Show_Record();
        }
        else if (Choice== 5)
        {
            cout << "Enter Roll Number of Student whose "
                    "record you want to Modify\n";
            cin >> Roll;
            Modify_record(Roll);
        }
        
        else if (Choice == 6) {
            cout<<"Exiting the program........"<<endl;
         for(int i=0; i<50; i++)
        {
            printf("--");
            Sleep(20);
        }
            exit(0);
        }
        else {
            cout << "Invalid Choice "
                 << "Try Again\n";
        }
    }
    while(Choice!=6);
    return 0;
}
