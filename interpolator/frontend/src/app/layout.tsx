import Link from 'next/link'
import './globals.css'

export const metadata = {
  title: '5D Interpolator',
  description: 'Machine Learning System for 5D Data Interpolation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <nav className="navbar">
          <div className="nav-container">
            <Link href="/" className="nav-logo">
              5D Interpolator
            </Link>
            <ul className="nav-menu">
              <li className="nav-item">
                <Link href="/upload" className="nav-link">
                  Upload
                </Link>
              </li>
              <li className="nav-item">
                <Link href="/train" className="nav-link">
                  Train
                </Link>
              </li>
              <li className="nav-item">
                <Link href="/predict" className="nav-link">
                  Predict
                </Link>
              </li>
            </ul>
          </div>
        </nav>
        <main className="main-content">
          {children}
        </main>
      </body>
    </html>
  )
}

