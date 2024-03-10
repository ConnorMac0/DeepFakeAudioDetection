import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container } from '@mui/material';
import './styles.css';
import ethanPhoto from './Images/ethanPhoto.png'

const Home = () => (
<Container>
  <div className='container'>
    <h1 className='font'>
      Real vs AI generated voice recognition
    </h1>
  </div>

  <div className='content'>
    <body>
      <p>
        A website designed to detect and mitigate vocal impersonation attacks by leveraging our 
        machine learning model, built with TensorFlow.
      </p>
    </body>
  </div>
</Container>
)

const About = ({ handleHomeClick }) => (
  <Container>
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          About
        </Typography>
        <Button color="inherit" component={Link} to="/" onClick={handleHomeClick}>Home</Button>
      </Toolbar>
    </AppBar>
    
    <div>
      <h1>About us</h1>
    </div>
    <div style={{ display: 'flex', alignItems: 'flex-start' }}>
      <div>
        <p>We are a team of 3 computer science students at the University of Oregon.</p>
        <h3>Ethan Hyde</h3>
        <p style={{ marginLeft: '20px'}}>
          Ethan's paragraph goes here
        </p>
        <h3>Connor MacLachlan</h3>
        <p style={{ marginLeft: '20px'}}>
          Connor's paragraph goes here
        </p>
        <h3>Kevin Truong</h3>
        <p style={{ marginLeft: '20px'}}>
          Kevin's paragraph goes here
        </p>
      </div>
      <div style={{ marginLeft: 'auto' }}>
        <img src={ethanPhoto} alt="Ethan's Photo" style={{ width: '100px' }} />
      </div>
    </div>
  </Container>
);



const Upload = ({ handleHomeClick }) => (
  <Container>
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Upload
        </Typography>
        <Button color="inherit" component={Link} to="/" onClick={handleHomeClick}>Home</Button>
      </Toolbar>
    </AppBar>
    <div>
      <h1 className='font'>File upload</h1>
      <body>
        <p>
          Choose an audio file to upload
        </p>
      </body>
    </div>
    
    <form>
      <input type="file" />
      <button type="submit">Upload</button>
    </form>
 
  </Container>
);

const Header = () => {
  const [isHeaderVisible, setIsHeaderVisible] = useState(true);

  const handleLinkClick = () => {
    setIsHeaderVisible(false);
  };

  const handleHomeClick = () => {
    setIsHeaderVisible(true);
  };

  return (
    <Router>
      {isHeaderVisible && (
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              CS 433 Project
            </Typography>
            <Button color="inherit" component={Link} to="/">Home</Button>
            <Button color="inherit" component={Link} to="/about" onClick={handleLinkClick}>About</Button>
            <Button color="inherit" component={Link} to="/upload" onClick={handleLinkClick}>Upload</Button>
          </Toolbar>
        </AppBar>
      )}

      <Routes>
        <Route path="/" element={<Home />} />
        <Route 
          path="/about" 
          element={<About handleHomeClick={handleHomeClick} />} />
        <Route
          path="/upload"
          element={<Upload handleHomeClick={handleHomeClick} />} />
      </Routes>
    </Router>
  );
};

export default Header;
