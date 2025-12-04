import React, { useState, useCallback } from 'react'
import {
  Box,
  Container,
  Heading,
  VStack,
  HStack,
  Button,
  Card,
  CardBody,
  Text,
  Badge,
  useToast,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Icon,
  Progress
} from '@chakra-ui/react'
import { useDropzone } from 'react-dropzone'
import { FiUpload, FiFile, FiTrash2 } from 'react-icons/fi'
import Layout from '@/components/Layout'

interface Dataset {
  id: number
  name: string
  size: number
  rows: number
  columns: string[]
  upload_time: string
  status: string
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const toast = useToast()

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    // Validate file type
    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload CSV or Excel files only.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
      return
    }

    // Validate file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
      toast({
        title: 'File too large',
        description: 'File size must be less than 100MB.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 200)

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (response.ok) {
        const result = await response.json()
        
        // Add the new dataset to the list
        setDatasets((prev) => [result.dataset, ...prev])
        
        toast({
          title: 'Upload successful',
          description: `${file.name} has been uploaded successfully.`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        })
      } else {
        const error = await response.json()
        throw new Error(error.error || 'Upload failed')
      }
    } catch (error) {
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }, [toast])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false,
    disabled: isUploading
  })

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const handleDelete = async (datasetId: number) => {
    try {
      const response = await fetch(`/api/datasets/${datasetId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        setDatasets((prev) => prev.filter(d => d.id !== datasetId))
        toast({
          title: 'Dataset deleted',
          description: 'Dataset has been successfully deleted.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        })
      } else {
        throw new Error('Failed to delete dataset')
      }
    } catch (error) {
      toast({
        title: 'Delete failed',
        description: 'Failed to delete dataset. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    }
  }

  return (
    <Layout>
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} align="stretch">
          <Box>
            <Heading mb={2}>Datasets</Heading>
            <Text color="gray.600">
              Upload and manage your datasets for machine learning experiments
            </Text>
          </Box>

          {/* Upload Section */}
          <Card>
            <CardBody>
              <VStack spacing={4}>
                <Box
                  {...getRootProps()}
                  border="2px dashed"
                  borderColor={isDragActive ? "brand.500" : "gray.300"}
                  borderRadius="md"
                  p={10}
                  textAlign="center"
                  cursor="pointer"
                  _hover={{ borderColor: "brand.500", bg: "brand.50" }}
                  bg={isDragActive ? "brand.50" : "white"}
                  transition="all 0.2s"
                  w="full"
                >
                  <input {...getInputProps()} />
                  <VStack spacing={4}>
                    <Icon as={FiUpload} boxSize={12} color="brand.500" />
                    <VStack spacing={2}>
                      <Text fontSize="lg" fontWeight="semibold">
                        {isDragActive ? 'Drop your file here' : 'Upload Dataset'}
                      </Text>
                      <Text color="gray.600">
                        Drag and drop your CSV or Excel file here, or click to browse
                      </Text>
                      <Text fontSize="sm" color="gray.500">
                        Supported formats: CSV, XLSX, XLS (Max 100MB)
                      </Text>
                    </VStack>
                  </VStack>
                </Box>
                
                {isUploading && (
                  <Box w="full">
                    <Text mb={2} fontSize="sm" color="gray.600">
                      Uploading... {uploadProgress}%
                    </Text>
                    <Progress value={uploadProgress} colorScheme="brand" />
                  </Box>
                )}
              </VStack>
            </CardBody>
          </Card>

          {/* Datasets List */}
          {datasets.length > 0 && (
            <Box>
              <Heading size="md" mb={4}>Your Datasets</Heading>
              <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                {datasets.map((dataset) => (
                  <Card key={dataset.id}>
                    <CardBody>
                      <VStack align="start" spacing={4}>
                        <HStack justify="space-between" w="full">
                          <HStack>
                            <Icon as={FiFile} color="brand.500" />
                            <Text fontWeight="semibold" noOfLines={1}>
                              {dataset.name}
                            </Text>
                          </HStack>
                          <Badge colorScheme={dataset.status === 'uploaded' ? 'green' : 'gray'}>
                            {dataset.status}
                          </Badge>
                        </HStack>

                        <SimpleGrid columns={2} spacing={4} w="full">
                          <Stat>
                            <StatLabel>Rows</StatLabel>
                            <StatNumber fontSize="md">{dataset.rows?.toLocaleString() || 'N/A'}</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel>Columns</StatLabel>
                            <StatNumber fontSize="md">{dataset.columns?.length || 'N/A'}</StatNumber>
                          </Stat>
                        </SimpleGrid>

                        <VStack align="start" spacing={1} w="full">
                          <Text fontSize="sm" color="gray.600">
                            Size: {formatFileSize(dataset.size)}
                          </Text>
                          <Text fontSize="sm" color="gray.600">
                            Uploaded: {formatDate(dataset.upload_time)}
                          </Text>
                        </VStack>

                        <HStack justify="space-between" w="full">
                          <Button size="sm" colorScheme="brand" variant="outline">
                            View Details
                          </Button>
                          <Button
                            size="sm"
                            colorScheme="red"
                            variant="ghost"
                            leftIcon={<FiTrash2 />}
                            onClick={() => handleDelete(dataset.id)}
                          >
                            Delete
                          </Button>
                        </HStack>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            </Box>
          )}

          {datasets.length === 0 && !isUploading && (
            <Card>
              <CardBody>
                <VStack spacing={4} py={8}>
                  <Icon as={FiFile} boxSize={16} color="gray.300" />
                  <Text color="gray.600" textAlign="center">
                    No datasets uploaded yet. Upload your first dataset to get started!
                  </Text>
                </VStack>
              </CardBody>
            </Card>
          )}
        </VStack>
      </Container>
    </Layout>
  )
}