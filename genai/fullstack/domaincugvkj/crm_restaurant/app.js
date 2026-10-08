const express = require('express');
const session = require('express-session');
const path = require('path');
const app = express();

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Session configuration
app.use(session({
    secret: 'restaurant-crm-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: { maxAge: 600000 } // 10 minutes
}));

// Set EJS as view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Hardcoded credentials
const VALID_USERNAME = 'admin';
const VALID_PASSWORD = 'password123';

// Store customer data in memory
let customers = [];

// =============================================
// ROUTES
// =============================================

// GET /login - Display login page
app.get('/login', (req, res) => {
    res.render('login', { error: null });
});

// GET / - Redirect to login
app.get('/', (req, res) => {
    res.redirect('/login');
});

// POST /home - Handle login form submission
app.post('/home', (req, res) => {
    const { username, password } = req.body;

    // Check hardcoded credentials
    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
        // Create session
        req.session.username = username;
        req.session.isLoggedIn = true;
        res.render('home', { username: username });
    } else {
        res.render('login', { error: 'Invalid username or password!' });
    }
});

// GET /surprise - Display surprise offer
app.get('/surprise', (req, res) => {
    // Check if session exists
    if (!req.session.isLoggedIn) {
        return res.redirect('/login');
    }
    res.render('surprise', { username: req.session.username });
});

// GET /customer - Display customer information in table
app.get('/customer', (req, res) => {
    // Check if session exists
    if (!req.session.isLoggedIn) {
        return res.redirect('/login');
    }
    const { age, mobile } = req.query;

    // Add new customer if age and mobile are provided
    if (age && mobile) {
        customers.push({ age: age, mobile: mobile });
    }

    res.render('customer', {
        customers: customers,
        username: req.session.username
    });
});

// GET /logout - Destroy session and redirect to login
app.get('/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            console.log('Error destroying session:', err);
        }
        customers = []; // Clear customer data
        res.redirect('/login');
    });
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Restaurant CRM Server running at http://localhost:${PORT}`);
    console.log(`Login at: http://localhost:${PORT}/login`);
    console.log(`Credentials: username = admin, password = password123`);
});
