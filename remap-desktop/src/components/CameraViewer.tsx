import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { CameraPose } from '../types'

function CameraPoints({ cameras }: { cameras: CameraPose[] }) {
  const points = cameras.map((camera) => {
    const matrix = camera.world_from_cam
    return [matrix[0]?.[3] ?? 0, matrix[1]?.[3] ?? 0, matrix[2]?.[3] ?? 0] as const
  })

  return (
    <>
      {points.map((position, index) => (
        <mesh key={`${position.join('_')}_${index}`} position={position}>
          <sphereGeometry args={[0.03, 12, 12]} />
          <meshStandardMaterial color="#8b5cf6" />
        </mesh>
      ))}
    </>
  )
}

export function CameraViewer({ cameras }: { cameras: CameraPose[] }) {
  if (cameras.length === 0) {
    return <div className="empty">Aucune caméra live disponible.</div>
  }

  return (
    <div className="viewer-wrap">
      <Canvas camera={{ position: [1.5, 1.5, 1.5], fov: 50 }}>
        <ambientLight intensity={0.7} />
        <pointLight position={[5, 5, 5]} intensity={1.2} />
        <axesHelper args={[0.5]} />
        <gridHelper args={[4, 12, '#334155', '#1e293b']} />
        <CameraPoints cameras={cameras} />
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  )
}
