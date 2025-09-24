import React from 'react'
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  HStack,
  Button,
  SimpleGrid,
  Card,
  CardBody,
  CardHeader,
  Icon,
  useColorModeValue
} from '@chakra-ui/react'
import { FiUpload, FiBarChart, FiCpu, FiTarget } from 'react-icons/fi'
import Link from 'next/link'
import Layout from '@/components/Layout'

export default function Home() {
  const cardBg = useColorModeValue('white', 'gray.800')

  return (
    <Layout>
      <Container maxW="container.xl" py={10}>
        <VStack spacing={10} align="center">
          <Box textAlign="center">
            <Heading size="2xl" mb={4} bgGradient="linear(to-r, brand.400, brand.600)" bgClip="text">
              Accurate 🎯
            </Heading>
            <Text fontSize="xl" color="gray.600" maxW="2xl">
              Comprehensive Machine Learning Platform - Upload, Analyze, Train, and Predict with ease
            </Text>
          </Box>

          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} w="full">
            <Link href="/datasets" passHref>
              <Card 
                bg={cardBg} 
                cursor="pointer" 
                _hover={{ transform: 'translateY(-4px)', shadow: 'lg' }}
                transition="all 0.2s"
              >
                <CardHeader>
                  <VStack>
                    <Icon as={FiUpload} boxSize={8} color="brand.500" />
                    <Heading size="md">Datasets</Heading>
                  </VStack>
                </CardHeader>
                <CardBody pt={0}>
                  <Text textAlign="center" color="gray.600">
                    Upload and manage your CSV/Excel datasets
                  </Text>
                </CardBody>
              </Card>
            </Link>

            <Link href="/eda" passHref>
              <Card 
                bg={cardBg} 
                cursor="pointer" 
                _hover={{ transform: 'translateY(-4px)', shadow: 'lg' }}
                transition="all 0.2s"
              >
                <CardHeader>
                  <VStack>
                    <Icon as={FiBarChart} boxSize={8} color="brand.500" />
                    <Heading size="md">EDA</Heading>
                  </VStack>
                </CardHeader>
                <CardBody pt={0}>
                  <Text textAlign="center" color="gray.600">
                    Exploratory Data Analysis and insights
                  </Text>
                </CardBody>
              </Card>
            </Link>

            <Link href="/training" passHref>
              <Card 
                bg={cardBg} 
                cursor="pointer" 
                _hover={{ transform: 'translateY(-4px)', shadow: 'lg' }}
                transition="all 0.2s"
              >
                <CardHeader>
                  <VStack>
                    <Icon as={FiCpu} boxSize={8} color="brand.500" />
                    <Heading size="md">Training</Heading>
                  </VStack>
                </CardHeader>
                <CardBody pt={0}>
                  <Text textAlign="center" color="gray.600">
                    Train ML models with 10+ algorithms
                  </Text>
                </CardBody>
              </Card>
            </Link>

            <Link href="/prediction" passHref>
              <Card 
                bg={cardBg} 
                cursor="pointer" 
                _hover={{ transform: 'translateY(-4px)', shadow: 'lg' }}
                transition="all 0.2s"
              >
                <CardHeader>
                  <VStack>
                    <Icon as={FiTarget} boxSize={8} color="brand.500" />
                    <Heading size="md">Prediction</Heading>
                  </VStack>
                </CardHeader>
                <CardBody pt={0}>
                  <Text textAlign="center" color="gray.600">
                    Make predictions with trained models
                  </Text>
                </CardBody>
              </Card>
            </Link>
          </SimpleGrid>

          <Box textAlign="center">
            <Text color="gray.600" mb={4}>
              Complete ML workflow from data upload to model deployment
            </Text>
            <HStack justify="center">
              <Link href="/datasets">
                <Button colorScheme="brand" size="lg">
                  Get Started
                </Button>
              </Link>
            </HStack>
          </Box>
        </VStack>
      </Container>
    </Layout>
  )
}