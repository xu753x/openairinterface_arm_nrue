

#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"

// ----- Old implementation ----
uint8_t **crc24c_generator_matrix(uint16_t payloadSizeBits){

	uint8_t crcPolynomialPattern[25] = {1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1};
	// 1011 0010 1011 0001 0001 0111 D^24 + D^23 + D^21 + D^20 + D^17 + D^15 + D^13 + D^12 + D^8 + D^4 + D^2 + D + 1
	uint8_t crcPolynomialSize = 24;
	uint8_t temp1[crcPolynomialSize], temp2[crcPolynomialSize];

	uint8_t **crc_generator_matrix = malloc(payloadSizeBits * sizeof(uint8_t *));
	if (crc_generator_matrix)
	  for (int i = 0; i < payloadSizeBits; i++)
		  crc_generator_matrix[i] = malloc(crcPolynomialSize * sizeof(uint8_t));

	for (int i = 0; i < crcPolynomialSize; i++) crc_generator_matrix[payloadSizeBits-1][i]=crcPolynomialPattern[i+1];

	for (int i = payloadSizeBits-2; i >= 0; i--){
		for (int j = 0; j < crcPolynomialSize-1; j++) temp1[j]=crc_generator_matrix[i+1][j+1];

		temp1[crcPolynomialSize-1]=0;

		for (int j = 0; j < crcPolynomialSize; j++)
			temp2[j]=crc_generator_matrix[i+1][0]*crcPolynomialPattern[j+1];

		for (int j = 0; j < crcPolynomialSize; j++){
			if(temp1[j]+temp2[j] == 1)
				crc_generator_matrix[i][j]=1;
			else
				crc_generator_matrix[i][j]=0;
		}
	}
	return crc_generator_matrix;
}

uint8_t **crc11_generator_matrix(uint16_t payloadSizeBits){

	uint8_t crcPolynomialPattern[12] = {1,1,1,0,0,0,1,0,0,0,0,1};
	// 1110 0010 0001 D^11 + D^10 + D^9 + D^5 + 1
	uint8_t crcPolynomialSize = 11;
	uint8_t temp1[crcPolynomialSize], temp2[crcPolynomialSize];

	uint8_t **crc_generator_matrix = malloc(payloadSizeBits * sizeof(uint8_t *));
	if (crc_generator_matrix)
	  for (int i = 0; i < payloadSizeBits; i++)
		  crc_generator_matrix[i] = malloc(crcPolynomialSize * sizeof(uint8_t));

	for (int i = 0; i < crcPolynomialSize; i++) crc_generator_matrix[payloadSizeBits-1][i]=crcPolynomialPattern[i+1];

	for (int i = payloadSizeBits-2; i >= 0; i--){
		for (int j = 0; j < crcPolynomialSize-1; j++)
			temp1[j]=crc_generator_matrix[i+1][j+1];

		temp1[crcPolynomialSize-1]=0;

		for (int j = 0; j < crcPolynomialSize; j++)
			temp2[j]=crc_generator_matrix[i+1][0]*crcPolynomialPattern[j+1];

		for (int j = 0; j < crcPolynomialSize; j++){
			if(temp1[j]+temp2[j] == 1)
				crc_generator_matrix[i][j]=1;
			else
				crc_generator_matrix[i][j]=0;
		}
	}

	return crc_generator_matrix;
}

uint8_t **crc6_generator_matrix(uint16_t payloadSizeBits){

	uint8_t crcPolynomialPattern[7] = {1,1,0,0,0,0,1};
	// 0110 0001 D^6 + D^5 + 1
	uint8_t crcPolynomialSize = 6;
	uint8_t temp1[crcPolynomialSize], temp2[crcPolynomialSize];
	uint8_t **crc_generator_matrix = malloc(payloadSizeBits * sizeof(uint8_t *));

	if (crc_generator_matrix)
	  for (int i = 0; i < payloadSizeBits; i++)
		  crc_generator_matrix[i] = malloc(crcPolynomialSize * sizeof(uint8_t));

	for (int i = 0; i < crcPolynomialSize; i++)
		crc_generator_matrix[payloadSizeBits-1][i]=crcPolynomialPattern[i+1];

	for (int i = payloadSizeBits-2; i >= 0; i--){
		for (int j = 0; j < crcPolynomialSize-1; j++)
			temp1[j]=crc_generator_matrix[i+1][j+1];

		temp1[crcPolynomialSize-1]=0;

		for (int j = 0; j < crcPolynomialSize; j++)
			temp2[j]=crc_generator_matrix[i+1][0]*crcPolynomialPattern[j+1];

		for (int j = 0; j < crcPolynomialSize; j++){
			if(temp1[j]+temp2[j] == 1)
				crc_generator_matrix[i][j]=1;
			else
				crc_generator_matrix[i][j]=0;
		}
	}

	return crc_generator_matrix;
}
