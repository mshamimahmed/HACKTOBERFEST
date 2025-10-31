import React from 'react'
import { Paper, Typography, TextField, Button, Box, Table, TableBody, TableCell, TableHead, TableRow, Dialog, DialogTitle, DialogContent, DialogActions, CircularProgress, Chip, Grid, Card, CardContent } from '@mui/material'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { useMutation } from '@tanstack/react-query'
import api from '../../services/api'

export default function Researcher(){
  const [text, setText] = React.useState('')
  const [results, setResults] = React.useState([])
  const [hypotheses, setHypotheses] = React.useState([])
  const [explanation, setExplanation] = React.useState('')
  const [queryId, setQueryId] = React.useState(null)
  const [confirmOpen, setConfirmOpen] = React.useState(false)
  const [pendingRow, setPendingRow] = React.useState(null) // row object awaiting confirmation
  const [expandedId, setExpandedId] = React.useState(null) // disease_id currently expanded
  const [compositionById, setCompositionById] = React.useState({}) // disease_id -> composition payload

  const COLORS = ['#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6','#84CC16']

  // persistent search history (last 10 unique queries)
  const [history, setHistory] = React.useState([])
  React.useEffect(()=>{
    try {
      const raw = localStorage.getItem('sr_history_v1')
      if (raw) setHistory(JSON.parse(raw))
    } catch {}
  }, [])
  const saveHistory = (arr) => {
    setHistory(arr)
    try { localStorage.setItem('sr_history_v1', JSON.stringify(arr)) } catch {}
  }

  const searchMutation = useMutation({
    mutationFn: async (payload) => {
      const res = await api.post('/api/researcher/search', payload)
      return res.data
    },
    onSuccess: (data) => {
      setQueryId(data.query_id || null)
      setResults(Array.isArray(data.matches) ? data.matches : [])
      setHypotheses(Array.isArray(data.hypotheses) ? data.hypotheses : [])
      setExplanation(data.explanation || '')
      setExpandedId(null)
      setCompositionById({})
      // update history
      const q = (text || '').trim()
      if (q) {
        const next = [q, ...history.filter(h => h !== q)].slice(0, 10)
        saveHistory(next)
      }
    }
  })

  const compositionMutation = useMutation({
    mutationFn: async ({ disease_id, selected_row_id, selected_symptoms }) => {
      const res = await api.post('/api/researcher/generate-composition', { disease_id, selected_row_id, selected_symptoms })
      return res.data
    },
    onSuccess: (data) => {
      if (data && data.disease_id) {
        setCompositionById(prev => ({ ...prev, [data.disease_id]: data }))
      }
    }
  })

  const onSubmit = (e) => {
    e?.preventDefault?.()
    const payload = { text: text }
    searchMutation.mutate(payload)
  }

  const onRowClick = (row, idx) => {
    // Open confirm dialog; do not expand yet
    setPendingRow({ ...row, _rowId: `match-${idx}` })
    setConfirmOpen(true)
  }

  const handleConfirm = (answer) => {
    const row = pendingRow
    setConfirmOpen(false)
    setPendingRow(null)
    if (answer !== 'yes' || !row) return
    // Collapse previous, expand this one and fetch composition
    setExpandedId(row.disease_id)
    compositionMutation.mutate({
      disease_id: row.disease_id,
      selected_row_id: row._rowId,
      selected_symptoms: row.matched_symptom_snippet || ''
    })
  }

  const downloadJSON = (obj, filename) => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(obj, null, 2))
    const a = document.createElement('a')
    a.setAttribute('href', dataStr)
    a.setAttribute('download', filename)
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  // chart data derived from results or hypotheses
  const chartData = (results.length>0 ? results : hypotheses).slice(0, 6).map((r, i) => ({
    name: r.disease_name?.slice(0, 14) || `Hypo ${i+1}`,
    score: Math.round(((r.match_score||r.score||0)*100))
  }))

  const handleHistoryClick = (q) => {
    setText(q)
    searchMutation.mutate({ text: q })
  }
  const clearHistory = () => saveHistory([])

  return (
    <Paper className="accent-card" sx={{ p: 3 }}>
      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>Researcher</Typography>
      <Typography variant="body2" className="muted">Paste symptoms, get disease matches and suggested compositions.</Typography>

      <Box component="form" onSubmit={onSubmit} sx={{ mt: 2, display: 'grid', gap: 2 }}>
        <TextField
          label="Symptoms"
          placeholder="Paste symptoms here (e.g., fever, cough, severe headache...)"
          value={text}
          onChange={(e)=>setText(e.target.value)}
          multiline
          minRows={3}
        />
        <Box sx={{ display:'flex', gap:1, alignItems:'center' }}>
          <Button type="submit" variant="contained" disabled={searchMutation.isPending}>Search</Button>
          {searchMutation.isPending && <CircularProgress size={20} />}
          {!searchMutation.isPending && (
            <Typography variant="body2" color="text.secondary">
              {results.length} match(es){hypotheses.length>0 ? `, ${hypotheses.length} hypothesis(es)` : ''}
            </Typography>
          )}
        </Box>
      </Box>

      {explanation && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">{explanation}</Typography>
        </Box>
      )}

      {/* Sidebar + charts */
      }
      <Grid container spacing={2} sx={{ mt: 2 }}>
        {/* Left: search history */}
        <Grid item xs={12} md={3}>
          <Card variant="outlined" sx={{ borderRadius: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>Search History</Typography>
              {history.length === 0 ? (
                <Typography className="muted">No history yet. Your last 10 queries will appear here.</Typography>
              ) : (
                <Box sx={{ display:'grid', gap: 1 }}>
                  {history.map((h, i)=> (
                    <Button key={i} size="small" variant="outlined" onClick={()=>handleHistoryClick(h)} sx={{ justifyContent:'flex-start', textTransform:'none' }}>
                      {h.length>60 ? (h.slice(0,60)+"…") : h}
                    </Button>
                  ))}
                  <Box sx={{ display:'flex', justifyContent:'flex-end' }}>
                    <Button size="small" color="secondary" onClick={clearHistory}>Clear</Button>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Middle: bar chart */}
        <Grid item xs={12} md={5}>
          <Card variant="outlined" sx={{ borderRadius: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>Top Matches</Typography>
              <Box sx={{ height: 240 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-20} height={50} dy={10} />
                    <YAxis domain={[0,100]} tickFormatter={(v)=>`${v}%`} width={36} />
                    <Tooltip formatter={(v)=>`${v}%`} />
                    <Bar dataKey="score" radius={[4,4,0,0]}>
                      {chartData.map((_, i) => (<Cell key={`c-${i}`} fill={COLORS[i % COLORS.length]} />))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        {/* Right: composition pie */}
        <Grid item xs={12} md={4}>
          <Card variant="outlined" sx={{ borderRadius: 2, height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>Composition (selected)</Typography>
              <Box sx={{ height: 240, display:'flex', alignItems:'center', justifyContent:'center' }}>
                {expandedId && compositionById[expandedId] && (compositionById[expandedId].composition||[]).length>0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={(compositionById[expandedId].composition||[]).map((c,i)=>({ name: c.ingredient, value: Math.round((c.confidence||0)*100) || 50 }))} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                        {(compositionById[expandedId].composition||[]).map((_, i)=> <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip formatter={(v)=>`${v}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography className="muted">Select a match to view composition pie chart.</Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Match Score</TableCell>
              <TableCell>Disease Name</TableCell>
              <TableCell>Matched Symptoms</TableCell>
              <TableCell>Existing Drugs</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(results.length>0 ? results : hypotheses.map(h => ({
              disease_id: h.hypothesis_id,
              disease_name: h.title,
              match_score: h.score || 0.7,
              matched_symptom_snippet: h.reasoning,
              existing_drugs: [],
            })) ).map((row, idx) => {
              const isExpanded = expandedId === row.disease_id
              const comp = compositionById[row.disease_id]
              return (
                <React.Fragment key={`${row.disease_id}-${idx}`}>
                  <TableRow hover onClick={()=>onRowClick(row, idx)} sx={{ cursor:'pointer' }}>
                    <TableCell>{Math.round((row.match_score||0)*100)}%</TableCell>
                    <TableCell>{row.disease_name}</TableCell>
                    <TableCell>{row.matched_symptom_snippet}</TableCell>
                    <TableCell>{Array.isArray(row.existing_drugs) ? row.existing_drugs.length : 0}</TableCell>
                  </TableRow>
                  {isExpanded && (
                    <TableRow>
                      <TableCell colSpan={4}>
                        {compositionMutation.isPending && !comp && (
                          <Box sx={{ py:2, display:'flex', alignItems:'center', gap:1 }}>
                            <CircularProgress size={20} />
                            <Typography>Generating composition…</Typography>
                          </Box>
                        )}
                        {comp && (
                          <Box>
                            <Box sx={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                              <Typography variant="subtitle1">Suggested Drug Composition — Auto-generated (research only)</Typography>
                              <Button size="small" onClick={()=>downloadJSON(comp, `${row.disease_id}-composition.json`)}>Download JSON</Button>
                            </Box>
                            <Box sx={{ mt:1 }}>
                              {(comp.composition || []).length === 0 ? (
                                <Typography color="text.secondary">No composition available.</Typography>
                              ) : (
                                <Table size="small">
                                  <TableHead>
                                    <TableRow>
                                      <TableCell>Ingredient</TableCell>
                                      <TableCell>From Drugs</TableCell>
                                      <TableCell>Suggested Amount</TableCell>
                                      <TableCell>Rationale</TableCell>
                                      <TableCell>Confidence</TableCell>
                                    </TableRow>
                                  </TableHead>
                                  <TableBody>
                                    {comp.composition.map((c,i)=> (
                                      <TableRow key={i}>
                                        <TableCell>{c.ingredient}</TableCell>
                                        <TableCell>{Array.isArray(c.from_drugs) ? c.from_drugs.join(', ') : ''}</TableCell>
                                        <TableCell>{c.suggested_amount}</TableCell>
                                        <TableCell>{c.rationale}</TableCell>
                                        <TableCell><Chip label={`${Math.round((c.confidence||0)*100)}%`} size="small" /></TableCell>
                                      </TableRow>
                                    ))}
                                  </TableBody>
                                </Table>
                              )}
                            </Box>
                            {comp.notes && <Typography variant="caption" color="text.secondary">{comp.notes}</Typography>}
                          </Box>
                        )}
                      </TableCell>
                    </TableRow>
                  )}
                </React.Fragment>
              )}
            )}
          </TableBody>
        </Table>
      </Box>

      <Dialog open={confirmOpen} onClose={()=>setConfirmOpen(false)}>
        <DialogTitle>Confirm</DialogTitle>
        <DialogContent>
          <Typography>
            This looks like a match (score: {pendingRow ? Math.round((pendingRow.match_score||0)*100) : 0}%). Would you like me to suggest a drug composition for this disease?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={()=>handleConfirm('no')}>No</Button>
          <Button onClick={()=>handleConfirm('cancel')}>Cancel</Button>
          <Button onClick={()=>handleConfirm('yes')} variant="contained">Yes</Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ mt: 3 }}>
        <Typography variant="caption" color="text.secondary">
          Only one row can be expanded at a time. Selecting a different row will close the previous one.
        </Typography>
      </Box>
    </Paper>
  )
}
