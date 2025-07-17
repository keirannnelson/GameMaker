
const firebaseConfig = {
    apiKey: "AIzaSyCX1cjeiVjnQjjExt-m1n8baTldB6kkKww",
    authDomain: "seo-tech-fd7cd.firebaseapp.com",
    projectId: "seo-tech-fd7cd",
    storageBucket: "seo-tech-fd7cd.firebasestorage.app",
    messagingSenderId: "344092136940",
    appId: "1:344092136940:web:9410e8811424b602b85cc9",
    measurementId: "G-5KSSNMJP7V"
  };
  
  firebase.initializeApp(firebaseConfig);
  

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
  
