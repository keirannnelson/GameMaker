// 🔁 Replace with your Firebase Web SDK config
const firebaseConfig = {
    apiKey: "AIzaSyDf1Xkg0_kqM4ea1Jkv4k4Iekjd3LOfunQ",
    authDomain: "seotechdev-16059.firebaseapp.com",
    projectId: "seotechdev-16059",
    storageBucket: "seotechdev-16059.firebasestorage.app",
    messagingSenderId: "511975477729",
    appId: "1:511975477729:web:69ff902a46c946288e42a0",
    measurementId: "G-KXWFEV10B3"
  };
  
  firebase.initializeApp(firebaseConfig);
  
  // 🔐 Signup Function
  async function signupUser() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
  
    try {
      await firebase.auth().createUserWithEmailAndPassword(email, password);
      alert("Signup successful! Redirecting to login...");
      window.location.href = "/login";
    } catch (error) {
      alert("Signup failed: " + error.message);
    }
  }
  
  // 🔐 Login Function
  async function loginUser() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
  
    try {
      const userCredential = await firebase.auth().signInWithEmailAndPassword(email, password);
      const idToken = await userCredential.user.getIdToken();
  
      // Send ID token to Flask
      const response = await fetch("/sessionLogin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ idToken }),
      });
  
      if (response.ok) {
        window.location.href = "/";
      } else {
        alert("Login failed. Please try again.");
      }
    } catch (error) {
      alert("Login error: " + error.message);
    }
  }
  