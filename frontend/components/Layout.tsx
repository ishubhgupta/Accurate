import React from 'react'
import {
  Box,
  Flex,
  HStack,
  VStack,
  Heading,
  Spacer,
  Button,
  useColorModeValue,
  Container,
  Text
} from '@chakra-ui/react'
import { FiHome, FiDatabase, FiBarChart, FiCpu, FiTarget } from 'react-icons/fi'
import Link from 'next/link'

interface LayoutProps {
  children: React.ReactNode
}

const NavButton = ({ href, icon, label }: { href: string; icon: any; label: string }) => {
  return (
    <Link href={href} passHref>
      <Button
        leftIcon={icon}
        variant="ghost"
        colorScheme="brand"
        size="sm"
      >
        {label}
      </Button>
    </Link>
  )
}

export default function Layout({ children }: LayoutProps) {
  const bgColor = useColorModeValue('gray.50', 'gray.900')
  const headerBg = useColorModeValue('white', 'gray.800')

  return (
    <Box minH="100vh" bg={bgColor}>
      <Box bg={headerBg} boxShadow="sm" position="sticky" top={0} zIndex={10}>
        <Container maxW="container.xl">
          <Flex h="64px" align="center">
            <Link href="/">
              <Heading size="md" cursor="pointer" color="brand.500">
                Accurate 🎯
              </Heading>
            </Link>
            
            <Spacer />
            
            <HStack spacing={4}>
              <NavButton href="/" icon={<FiHome />} label="Home" />
              <NavButton href="/datasets" icon={<FiDatabase />} label="Datasets" />
              <NavButton href="/eda" icon={<FiBarChart />} label="EDA" />
              <NavButton href="/training" icon={<FiCpu />} label="Training" />
              <NavButton href="/prediction" icon={<FiTarget />} label="Prediction" />
            </HStack>
          </Flex>
        </Container>
      </Box>
      
      <Box as="main">
        {children}
      </Box>
      
      <Box as="footer" bg={headerBg} py={8} mt={16}>
        <Container maxW="container.xl">
          <VStack spacing={2}>
            <Text fontSize="sm" color="gray.600">
              © 2024 Accurate ML Platform. Making machine learning accessible.
            </Text>
          </VStack>
        </Container>
      </Box>
    </Box>
  )
}