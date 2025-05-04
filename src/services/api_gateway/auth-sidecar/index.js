// auth-sidecar/index.js
const express = require('express');
const { ClerkExpressRequireAuth } = require('@clerk/clerk-sdk-node');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS.split(','),
  credentials: true
}));

// JWT verification endpoint
app.post('/verify', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    // Verify JWT using Clerk's public key
    const decoded = jwt.verify(token, process.env.CLERK_JWT_PUBLIC_KEY, {
      algorithms: ['RS256']
    });

    // Check if token is expired
    const now = Math.floor(Date.now() / 1000);
    if (decoded.exp && decoded.exp < now) {
      return res.status(401).json({ error: 'Token expired' });
    }

    // Add user info to headers
    res.setHeader('X-User-Id', decoded.sub);
    res.setHeader('X-User-Role', decoded.role || 'user');

    // Return success
    return res.status(200).json({ verified: true });
  } catch (error) {
    console.error('Token verification error:', error);
    return res.status(401).json({ error: 'Invalid token' });
  }
});

// Role-based authorization endpoint
app.post('/authorize', async (req, res) => {
  try {
    const userRole = req.headers['x-user-role'];
    const requiredRole = req.headers['x-required-role'];

    if (!userRole) {
      return res.status(401).json({ error: 'No user role provided' });
    }

    // Role hierarchy: admin > developer > viewer
    const roleValues = {
      'admin': 3,
      'developer': 2,
      'viewer': 1
    };

    // Check if user has sufficient permissions
    if (roleValues[userRole] >= roleValues[requiredRole]) {
      return res.status(200).json({ authorized: true });
    } else {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
  } catch (error) {
    console.error('Authorization error:', error);
    return res.status(500).json({ error: 'Authorization failed' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Auth sidecar running on port ${PORT}`);
});