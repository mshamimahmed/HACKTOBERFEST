import { CssBaseline, Container, Typography, Box, AppBar, Toolbar } from '@mui/material'
import { Routes, Route } from 'react-router-dom'
import Researcher from './routes/Researcher/Researcher'

export default function App() {
  return (
    <>
      <CssBaseline />
      <AppBar position="sticky" color="transparent" elevation={0} sx={{ backdropFilter: 'blur(6px)' }}>
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>AI-Driven Drug Repurposing</Typography>
          <Typography variant="body2" color="text.secondary">Research Preview</Typography>
        </Toolbar>
      </AppBar>

      {/* Gradient hero */}
      <Box className="hero-gradient" sx={{ py: 6, mb: 4 }}>
        <Container maxWidth="lg">
          <Typography variant="h4" sx={{ fontWeight: 800, mb: 1 }}>Researcher — Symptom-to-Disease Matching</Typography>
          <Typography variant="body1" color="text.secondary">
            Suggested drug compositions are auto-generated for research/testing only and are not clinical advice.
          </Typography>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ pb: 8 }}>
        <Routes>
          <Route path="/" element={<Researcher />} />
          <Route path="/researcher" element={<Researcher />} />
        </Routes>
      </Container>
    </>
  )
}
