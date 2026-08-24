import { useEffect, useState, useRef } from 'react'
import type { FormEvent } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import './App.css'

type Employee = Record<string, string | number>
type Prediction = { attrition_probability: number; prediction: 'Yes' | 'No'; risk_level: 'High' | 'Medium' | 'Low'; threshold_used: number }

const initialEmployee: Employee = { Age: 35, BusinessTravel: 'Travel_Rarely', DailyRate: 800, Department: 'Research & Development', DistanceFromHome: 5, Education: 3, EducationField: 'Life Sciences', EnvironmentSatisfaction: 3, Gender: 'Male', HourlyRate: 65, JobInvolvement: 3, JobLevel: 2, JobRole: 'Research Scientist', JobSatisfaction: 3, MaritalStatus: 'Single', MonthlyIncome: 5000, MonthlyRate: 14000, NumCompaniesWorked: 2, OverTime: 'No', PercentSalaryHike: 13, PerformanceRating: 3, RelationshipSatisfaction: 3, StockOptionLevel: 1, TotalWorkingYears: 10, TrainingTimesLastYear: 3, WorkLifeBalance: 3, YearsAtCompany: 5, YearsInCurrentRole: 3, YearsSinceLastPromotion: 1, YearsWithCurrManager: 3 }
const options: Record<string, string[]> = { BusinessTravel: ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], Department: ['Sales', 'Research & Development', 'Human Resources'], EducationField: ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], Gender: ['Male', 'Female'], JobRole: ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], MaritalStatus: ['Single', 'Married', 'Divorced'], OverTime: ['Yes', 'No'] }
const API_BASE = import.meta.env.VITE_API_URL || '/api'

async function health() { const response = await fetch(`${API_BASE}/`); if (!response.ok) throw new Error(`Health check failed (${response.status})`); return response.json() as Promise<{ model_ready: boolean }> }
async function predict(employee: Employee) { const response = await fetch(`${API_BASE}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(employee) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || `Prediction failed (${response.status})`); return body as Prediction }

// Custom Pirate Engraving Style SVGs with specific hover interactive motions
const CompassIcon = () => (
  <svg className="pirate-icon compass-needle-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="32" cy="32" r="26" />
    <circle cx="32" cy="32" r="6" />
    <path d="M32 6v6M32 52v6M6 32h6M52 32h6" />
    <path d="M32 26l6 6-6 16-6-16z" fill="currentColor" opacity="0.8" />
    <path d="M32 38l-4-6 4-16 4 6z" stroke="var(--gold)" fill="var(--gold)" />
  </svg>
)

const SwordsIcon = () => (
  <svg className="pirate-icon sword-slash-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 52l36-36M40 12l12 12" />
    <path d="M44 20L20 44" strokeWidth="3" />
    <path d="M14 46l4 4" />
    <path d="M52 12l-6 6" />
    <path d="M52 12L40 16" />
    <path d="M52 12l-4 12" />
    {/* Crossed sword 2 */}
    <path d="M52 52L16 16M24 12L12 24" />
    <path d="M20 20l24 24" strokeWidth="3" stroke="var(--gold)" />
    <path d="M50 46l-4 4" />
    <path d="M12 12l6 6" />
  </svg>
)

const SkullIcon = () => (
  <svg className="pirate-icon skull-glow-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M32 8c-10 0-16 6-16 16 0 7 3 13 8 16v8c0 3 2 5 5 5h6c3 0 5-2 5-5v-8c5-3 8-9 8-16 0-10-6-16-16-16z" />
    <circle cx="24" cy="24" r="4" fill="currentColor" />
    <circle cx="40" cy="24" r="4" fill="currentColor" />
    <path d="M32 30v4M28 42h8M30 48h4" />
    <path d="M10 54l8-8M54 54l-8-8M10 10l8 8M54 10l-8 8" strokeWidth="1.5" opacity="0.6" />
  </svg>
)

const AnchorIcon = () => (
  <svg className="pirate-icon anchor-bob-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="32" cy="10" r="4" />
    <path d="M32 14v30M24 22h16" />
    <path d="M12 36c0 12 10 20 20 20s20-8 20-20M8 32l4 4 4-4M56 32l-4 4-4-4" />
  </svg>
)

const FlagIcon = () => (
  <svg className="pirate-icon flag-wave-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 8v48M12 12c10-4 18 4 28 0s16-2 16-2v24c0 0-8 2-16 2s-18-8-28-4" fill="currentColor" opacity="0.2" />
    <path d="M26 22a2 2 0 100-4 2 2 0 000 4z" fill="var(--gold)" stroke="var(--gold)" />
    <path d="M32 20h2" stroke="var(--gold)" />
  </svg>
)

const WheelIcon = () => (
  <svg className="pirate-icon wheel-spin-anim" width="74" height="74" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="32" cy="32" r="16" />
    <circle cx="32" cy="32" r="5" fill="currentColor" />
    <path d="M32 4v12M32 48v12M4 32h12M48 32h12M12 12l9 9M43 43l9 9M12 52l9-9M43 21l9-9" />
  </svg>
)

const CustomSelectArrow = () => (
  <svg className="select-needle" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v14M7 12l5 5 5-5" />
  </svg>
)

function Field({ label, name, value, update, min, max, choices, onFocus }: { label: string; name: string; value: string | number; update: (name: string, value: string | number) => void; min?: number; max?: number; choices?: string[]; onFocus?: () => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      {choices ? (
        <span className="select-wrap">
          <select value={value} onChange={(event) => update(name, event.target.value)} onFocus={onFocus}>
            {choices.map((choice) => (
              <option key={choice} value={choice}>{choice.replace(/_/g, ' ')}</option>
            ))}
          </select>
          <CustomSelectArrow />
        </span>
      ) : (
        <input type="number" value={value} min={min} max={max} onChange={(event) => update(name, event.target.value === '' ? '' : Number(event.target.value))} onFocus={onFocus} />
      )}
    </label>
  )
}

function FieldGroup({ index, title, copy, icon: Icon, active, children }: { index: string; title: string; copy: string; icon: React.ComponentType; active: boolean; children: React.ReactNode }) {
  return (
    <motion.section className={`field-group ${active ? 'active-group' : ''}`} initial={{ opacity: 0, y: 22, filter: 'blur(6px)' }} whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }} viewport={{ once: true, amount: .15 }} transition={{ duration: .55 }}>
      <div className="group-title">
        <span className="group-marker">{index}</span>
        <div className="group-icon-wrap">
          <Icon />
        </div>
        <div>
          <p className="eyebrow group-name">{title}</p>
          <small className="group-copy">{copy}</small>
        </div>
      </div>
      <div className="field-grid">{children}</div>
    </motion.section>
  )
}

function CountUp({ value }: { value: number }) {
  const [count, setCount] = useState(0)
  useEffect(() => {
    let start = 0
    const end = value
    if (start === end) return
    const totalDuration = 1000
    const incrementTime = Math.max(Math.floor(totalDuration / end), 15)
    const timer = setInterval(() => {
      start += 1
      if (start >= end) {
        setCount(end)
        clearInterval(timer)
      } else {
        setCount(start)
      }
    }, incrementTime)
    return () => clearInterval(timer)
  }, [value])
  return <span>{count.toFixed(1)}</span>
}

function App() {
  const [theme, setTheme] = useState<'day' | 'night'>(() => (localStorage.getItem('grand-line-theme') as 'day' | 'night') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'night' : 'day'))
  const [employee, setEmployee] = useState(initialEmployee)
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState('')
  const [online, setOnline] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const [activeGroup, setActiveGroup] = useState<string>('A')
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  const predictionRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('grand-line-theme', theme)
  }, [theme])

  useEffect(() => {
    health().then((data) => setOnline(data.model_ready)).catch(() => setOnline(false))
    
    const onScroll = () => setScrolled(window.scrollY > 24)
    const onMouseMove = (e: MouseEvent) => {
      setMousePos({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20
      })
    }

    window.addEventListener('scroll', onScroll)
    window.addEventListener('mousemove', onMouseMove)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('mousemove', onMouseMove)
    }
  }, [])

  const update = (name: string, value: string | number) => setEmployee((current) => ({ ...current, [name]: value }))
  
  const number = (name: string, label: string, min?: number, max?: number, groupMarker = 'A') => (
    <Field name={name} label={label} value={employee[name]} update={update} min={min} max={max} onFocus={() => setActiveGroup(groupMarker)} />
  )
  const select = (name: string, label: string, groupMarker = 'A') => (
    <Field name={name} label={label} value={employee[name]} update={update} choices={options[name]} onFocus={() => setActiveGroup(groupMarker)} />
  )

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)

    // Animated Loading Sequence (synchronized with fake delay to match One Piece drama under 2s)
    for (let index = 1; index <= 3; index += 1) {
      setStage(index)
      await new Promise((resolve) => setTimeout(resolve, 500))
    }

    try {
      const response = await predict(employee)
      setResult(response)
      setTimeout(() => {
        predictionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 100)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The prediction service is unavailable.')
    } finally {
      setLoading(false)
      setStage(0)
    }
  }

  // Calculate completed sections metrics for Voyage Progress bar
  const checkFields = (fieldsList: string[]) => fieldsList.every(f => employee[f] !== undefined && employee[f] !== '')
  const stepA = checkFields(['Age', 'Gender', 'MaritalStatus', 'Education'])
  const stepB = checkFields(['Department', 'EducationField', 'JobRole', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany'])
  const stepC = checkFields(['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance'])
  const stepD = checkFields(['DailyRate', 'DistanceFromHome', 'HourlyRate', 'PercentSalaryHike'])

  const probability = result ? result.attrition_probability * 100 : 0
  const riskClass = result?.risk_level.toLowerCase() || 'low'

  return (
    <div className="app-shell" style={{ transform: `translate3d(${mousePos.x * 0.15}px, ${mousePos.y * 0.15}px, 0)` }}>
      {/* Immersive Animated Background Canvas */}
      <div className="energy-field">
        <div className="noise" />
        <div className="light-field light-one" />
        <div className="light-field light-two" />
        <div className="ribbon ribbon-one" />
        <div className="ribbon ribbon-two" />
        <div className="ribbon ribbon-three" />
        <div className="horizon" />
        
        {/* Layered Twilight Star Field */}
        <div className="star-layer star-back" />
        <div className="star-layer star-mid" />
        <div className="star-layer star-front" />
      </div>

      <header className={`topbar ${scrolled ? 'scrolled' : ''}`}>
        <a className="brand" href="#home">
          <span className="brand-orbit"><WheelIcon /></span>
          <span><b>GRAND LINE</b><small>EMPLOYEE ATTRITION INTELLIGENCE</small></span>
        </a>
        <nav>
          <a href="#home"><CompassIcon /> HOME</a>
          <a href="#employee"><SwordsIcon /> PREDICT</a>
          <a href="#analytics"><SkullIcon /> ANALYTICS</a>
          <a href="#insights"><AnchorIcon /> INSIGHTS</a>
          <a href="#system"><FlagIcon /> SYSTEM</a>
        </nav>
        <div className="top-actions">
          <button className="theme-toggle" onClick={() => setTheme(theme === 'day' ? 'night' : 'day')} aria-label="Toggle day and night theme">
            {theme === 'day' ? '☀' : '☾'}
          </button>
        </div>
      </header>

      <main>
        {/* Cinematic Grand Line Hero */}
        <section className="hero" id="home" style={{ transform: `translate3d(${mousePos.x * -0.2}px, ${mousePos.y * -0.2}px, 0)` }}>
          <div className="hero-center">
            <p className="eyebrow hero-label"><span /> THE GRAND LINE <span /></p>
            <h1>GRAND<br /><em>LINE</em></h1>
            <p className="hero-title">EMPLOYEE ATTRITION INTELLIGENCE</p>
            <p className="hero-copy">Navigate workforce data. Predict attrition. Discover risk. Keep your pirate crew united under one flag.</p>
            <div className="hero-actions">
              <a className="primary-cta" href="#employee">⚔ START PREDICTION</a>
              <a className="secondary-cta" href="#analytics">⚓ EXPLORE ANALYTICS</a>
            </div>
          </div>
          <div className="hero-readout">
            <span>01</span>
            <div className="readout-line" />
            <small>THE VOYAGE<br />BEGINS HERE</small>
          </div>
        </section>

        {/* Technical Signal Bar with nautical styling */}
        <section className="signal-bar">
          <span><b>MODEL</b> LOGISTIC REGRESSION</span>
          <span><b>FEATURES</b> 47 ENGINEERED</span>
          <span><b>THRESHOLD</b> 0.61</span>
          <span className="signal-message">🧭 Log Pose is calibrated. Ready to sail.</span>
        </section>

        {/* Voyage Progress Nautical Line */}
        <section className="voyage-progress-container">
          <div className="voyage-title">VOYAGE PROGRESS</div>
          <div className="voyage-track-wrapper">
            <div className="voyage-line">
              <div className={`voyage-fill ${stepD ? 'w-100' : stepC ? 'w-75' : stepB ? 'w-50' : stepA ? 'w-25' : ''}`} />
            </div>
            <div className="voyage-nodes">
              <div className={`voyage-node ${stepA ? 'reached' : ''}`} data-label="PERSONAL">
                <SkullIcon />
              </div>
              <div className={`voyage-node ${stepB ? 'reached' : ''}`} data-label="PROFESSIONAL">
                <SwordsIcon />
              </div>
              <div className={`voyage-node ${stepC ? 'reached' : ''}`} data-label="ENVIRONMENT">
                <AnchorIcon />
              </div>
              <div className={`voyage-node ${stepD ? 'reached' : ''}`} data-label="ADDITIONAL">
                <CompassIcon />
              </div>
            </div>
          </div>
        </section>

        {/* Employee Form (Grand Line Navigation Document) */}
        <section className="profile-section" id="employee">
          <div className="section-intro">
            <p className="eyebrow">01 / CREW MANIFEST</p>
            <h2>EMPLOYEE PROFILE</h2>
            <p>Update crew credentials for attrition forecasting analysis.</p>
          </div>

          <form onSubmit={submit}>
            <FieldGroup index="A" title="PERSONAL INFORMATION" copy="Basic identifiers and crew demographics" icon={SkullIcon} active={activeGroup === 'A'}>
              <>{number('Age', 'Age', 18, 65, 'A')}{select('Gender', 'Gender', 'A')}{select('MaritalStatus', 'Marital Status', 'A')}{number('Education', 'Education Level', 1, 5, 'A')}</>
            </FieldGroup>

            <FieldGroup index="B" title="PROFESSIONAL DETAILS" copy="Role assignments, rank and service tenure" icon={SwordsIcon} active={activeGroup === 'B'}>
              <>{select('Department', 'Department', 'B')}{select('EducationField', 'Education Field', 'B')}{select('JobRole', 'Job Role', 'B')}{number('JobLevel', 'Job Level', 1, 5, 'B')}{number('MonthlyIncome', 'Monthly Income (Gold Coins)', 0, undefined, 'B')}{number('YearsAtCompany', 'Years on this Voyage', 0, undefined, 'B')}{number('YearsInCurrentRole', 'Years in Current Duty', 0, undefined, 'B')}{number('YearsSinceLastPromotion', 'Years Since Last Promotion', 0, undefined, 'B')}{number('YearsWithCurrManager', 'Years Under Current Captain', 0, undefined, 'B')}{number('TotalWorkingYears', 'Total Pirate Career Years', 0, undefined, 'B')}</>
            </FieldGroup>

            <FieldGroup index="C" title="WORK ENVIRONMENT" copy="Wellbeing, morale, and voyage stability" icon={AnchorIcon} active={activeGroup === 'C'}>
              <>{number('JobSatisfaction', 'Job Satisfaction', 1, 4, 'C')}{number('EnvironmentSatisfaction', 'Environment Satisfaction', 1, 4, 'C')}{number('RelationshipSatisfaction', 'Relationship Satisfaction', 1, 4, 'C')}{number('WorkLifeBalance', 'Work-Life Balance', 1, 4, 'C')}{number('JobInvolvement', 'Duty Involvement', 1, 4, 'C')}{number('PerformanceRating', 'Performance Grade', 1, 4, 'C')}{select('OverTime', 'Overtime Duty Required', 'C')}{select('BusinessTravel', 'Sailing Frequency', 'C')}</>
            </FieldGroup>

            <FieldGroup index="D" title="COMPENSATION & GROWTH" copy="Additional compensation metrics and training signals" icon={CompassIcon} active={activeGroup === 'D'}>
              <>{number('DailyRate', 'Daily Rate', 0, undefined, 'D')}{number('DistanceFromHome', 'Distance from Home Port', 0, undefined, 'D')}{number('HourlyRate', 'Hourly Rate', 0, undefined, 'D')}{number('MonthlyRate', 'Monthly Rate', 0, undefined, 'D')}{number('NumCompaniesWorked', 'Previous Fleet Memberships', 0, undefined, 'D')}{number('PercentSalaryHike', 'Salary Increase Percentage', 0, undefined, 'D')}{number('StockOptionLevel', 'Share Level (Stock)', 0, 3, 'D')}{number('TrainingTimesLastYear', 'Combat Exercises Last Year', 0, undefined, 'D')}</>
            </FieldGroup>

            <div className="form-footer">
              <p>⚡ Direct connection established to the live Grand Line ML Predictor API.</p>
              <button className="predict-button" disabled={loading || !online}>
                {loading ? (
                  <>⚔ ANALYZING CREW SIGNS...</>
                ) : (
                  <>⚔ PREDICT ATTRITION</>
                )}
                <span className={loading ? 'progress' : ''} />
              </button>
            </div>
          </form>
        </section>

        {/* Prediction Results Section */}
        <section className="prediction-section" id="prediction" ref={predictionRef}>
          <div className="section-intro split">
            <div>
              <p className="eyebrow">02 / LOG POSE PROJECTION</p>
              <h2>THE LOG POSE HAS SPOKEN</h2>
              <p>Your result is generated by the trained prediction pipeline.</p>
            </div>
            <span className="endpoint"><CompassIcon /> POST /predict</span>
          </div>

          <AnimatePresence mode="wait">
            {loading && (
              <motion.div className="processing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="processing-orbit"><WheelIcon /></div>
                <p className="processing-text">
                  {['READING THE CREW DATA...', 'CALCULATING ATTRITION RISK...', 'THE LOG POSE HAS SPOKEN...'][stage - 1]}
                </p>
                <div className="processing-track">
                  <i style={{ width: `${stage * 33.33}%` }} />
                </div>
              </motion.div>
            )}

            {error && (
              <motion.div className="error-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <SwordsIcon />
                <div>
                  <b>Prediction Pipeline Error</b>
                  <p>{error}</p>
                </div>
              </motion.div>
            )}

            {result && (
              <motion.div className={`result-panel ${riskClass}`} initial={{ opacity: 0, scale: .96, filter: 'blur(8px)' }} animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}>
                {result.risk_level === 'High' ? (
                  /* Premium Wanted Poster prediction result for HIGH RISK */
                  <div className="wanted-poster-container">
                    <div className="wanted-poster">
                      <div className="wanted-header">WANTED</div>
                      <div className="wanted-image-area">
                        <SkullIcon />
                        <div className="red-stamp">HIGH RISK</div>
                      </div>
                      <div className="wanted-body">
                        <p className="wanted-label">EMPLOYEE ATTRITION RISK</p>
                        <div className="wanted-probability">
                          <CountUp value={probability} />%
                        </div>
                        <p className="wanted-attr-status">Prediction: {result.prediction === 'Yes' ? 'MUTINY LIKELY (LEAVING)' : 'LOYAL MEMBER (STAYING)'}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Standard Pirate Themed Clean UI Panel for Low/Medium Risks */
                  <div className="result-copy-block">
                    <span className="result-kicker">{result.risk_level === 'Medium' ? 'SIGNAL / UNSTABLE' : 'SIGNAL / SECURE'}</span>
                    <h3>{result.risk_level.toUpperCase()} RISK</h3>
                    <p>Model prediction: <strong>{result.prediction === 'Yes' ? 'Likely to mutiny' : 'Likely to stay'}</strong></p>
                  </div>
                )}

                {/* Risk Gauge responsive to mouse */}
                <div className="risk-gauge">
                  <div className="gauge-inner" style={{ '--progress': `${probability * 3.6}deg` } as React.CSSProperties}>
                    <strong>
                      <CountUp value={probability} />
                      <small>%</small>
                    </strong>
                    <span>probability</span>
                  </div>
                </div>

                <div className="result-facts">
                  <span>RISK LEVEL<b>{result.risk_level}</b></span>
                  <span>THRESHOLD<b>{result.threshold_used.toFixed(2)}</b></span>
                  <span>RESPONSE<b>LIVE MODEL API</b></span>
                </div>
              </motion.div>
            )}

            {!result && !loading && !error && (
              <div className="result-empty">
                <CompassIcon />
                <span>Awaiting your first prediction</span>
                <small>Complete the employee profile above to request the Log Pose verdict.</small>
              </div>
            )}
          </AnimatePresence>
        </section>

        {/* Crew Analytics */}
        <section className="analytics-section" id="analytics">
          <div className="section-intro">
            <p className="eyebrow">03 / CREW ANALYTICS</p>
            <h2>CREW ANALYTICS</h2>
            <p>An honest layout of what the ML system processes.</p>
          </div>
          <div className="metric-row">
            <article>
              <SkullIcon />
              <small>RAW INPUTS</small>
              <strong>30</strong>
              <span>Required Fields</span>
            </article>
            <article>
              <SwordsIcon />
              <small>ENGINEERED SIGNALS</small>
              <strong>47</strong>
              <span>Model Features</span>
            </article>
            <article>
              <AnchorIcon />
              <small>DECISION THRESHOLD</small>
              <strong>0.61</strong>
              <span>Validation Selected</span>
            </article>
            <article>
              <CompassIcon />
              <small>MODEL STATE</small>
              <strong>{online ? 'READY' : 'WAIT'}</strong>
              <span>Live API connection</span>
            </article>
          </div>
        </section>

        {/* Insights and Transparency */}
        <section className="insights-section" id="insights">
          <div>
            <p className="eyebrow">04 / TRANSPARENCY</p>
            <h2>Why this prediction?</h2>
            <p>The current API returns prediction, probability, risk level, and threshold only.</p>
          </div>
          <div className="insight-note">
            <CompassIcon />
            <div>
              <b>Feature explanations require backend support.</b>
              <span>No SHAP values or feature importance are fabricated here to ensure compliance with actual API output.</span>
            </div>
          </div>
        </section>

        {/* System status */}
        <section className="system-section" id="system">
          <div className="system-header">
            <div>
              <p className="eyebrow">05 / SHIP STATS</p>
              <h2>SYSTEM STATUS</h2>
            </div>
            <span className={`system-state ${online ? 'live' : ''}`}><i /> {online ? 'API ONLINE' : 'API OFFLINE'}</span>
          </div>
          <div className="system-grid">
            <div>
              <small>MODEL</small>
              <b>LOADED</b>
            </div>
            <div>
              <small>ENDPOINT</small>
              <b>/predict</b>
            </div>
            <div>
              <small>AUTHENTICATION</small>
              <b>NOT REQUIRED</b>
            </div>
            <div>
              <small>TRANSPORT</small>
              <b>HTTP / LOCAL</b>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <span className="footer-logo">
          <span className="brand-orbit"><WheelIcon /></span>
          <b>GRAND LINE</b>
        </span>
        <span>The voyage never ends.</span>
        <span className="footer-note">Machine learning for better crew decisions.</span>
      </footer>
    </div>
  )
}

export default App
