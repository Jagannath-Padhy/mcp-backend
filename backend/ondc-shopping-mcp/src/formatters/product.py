"""
Product formatter for MCP content

Handles formatting of ONDC product data for display in MCP.
"""

import sys
from typing import Dict, List, Optional, Any


class ProductFormatter:
    """Formats ONDC product data for MCP display"""
    
    def __init__(self):
        """Initialize product formatter"""
        pass
    
    async def format_search_results(self, result: Dict, image_formatter) -> List[Dict]:
        """
        Format search results with images.
        
        Args:
            result: Search result dictionary
            image_formatter: ImageFormatter instance for handling images
            
        Returns:
            List of MCP content items
        """
        # Handle new reranked search format
        if "search_results" in result and isinstance(result["search_results"], list):
            # New format with reranked results
            items = result["search_results"]
            message = result.get("message", "")
            search_type = result.get("search_type", "")
            total_results = result.get("total_results", len(items))
            
            # Extract query from message
            query = ""
            if "Found" in message and "products" in message:
                # Try to extract query if it's in the message
                parts = message.split("'")
                if len(parts) > 1:
                    query = parts[1]
            
            if not items:
                return [{
                    "type": "text", 
                    "text": "Sorry, I couldn't find any products. Please try a different search term."
                }]
                
        # Handle old format (fallback)
        elif "response" in result:
            response_data = result.get("response", {}).get("data", [])
            if not response_data:
                return [{
                    "type": "text", 
                    "text": "Sorry, I couldn't find any products. Please try a different search term."
                }]
            # Convert old format to new format
            items = []
            for item in response_data:
                items.append({"item": item})
            message = result.get("message", "")
            search_type = ""
            total_results = len(items)
            query = ""
        else:
            return [{
                "type": "text", 
                "text": "Sorry, I couldn't find any products. Please try a different search term."
            }]
        
        # Analyze source breakdown for transparency
        api_count = 0
        vector_count = 0
        hybrid_count = 0
        
        for result in items:
            if isinstance(result, dict) and "sources" in result:
                sources = result["sources"]
                if "api" in sources and "vector" in sources:
                    hybrid_count += 1
                elif "api" in sources:
                    api_count += 1
                elif "vector" in sources:
                    vector_count += 1
        
        # Start with enhanced header with source breakdown
        header_text = f" Found {total_results} products"
        if query:
            header_text += f" for '{query}'"
        
        # Add detailed search breakdown
        breakdown_parts = []
        if api_count > 0:
            breakdown_parts.append(f" {api_count} from MongoDB API")
        if vector_count > 0:
            breakdown_parts.append(f" {vector_count} from Vector DB")
        if hybrid_count > 0:
            breakdown_parts.append(f" {hybrid_count} from both sources")
        
        if breakdown_parts:
            header_text += f"\n **Search Breakdown:** {' | '.join(breakdown_parts)}"
            
        if search_type:
            header_text += f"\n **Search Type:** {search_type} (AI reranked)"
            
        header_text += "\n"
        
        content = [{"type": "text", "text": header_text}]
        
        # Add each product with images and text
        for i, result in enumerate(items[:5], 1):
            # Extract item from result (handles both formats)
            if isinstance(result, dict) and "item" in result:
                item = result["item"]
                # Add source info if available
                sources = result.get("sources", [])
                score = result.get("rerank_score", result.get("vector_score", 0))
            else:
                item = result
                sources = []
                score = 0
            
            product_content = await self._build_product_content(item, i, image_formatter, sources, score)
            content.extend(product_content)
        
        # Add footer
        footer_parts = []
        if len(items) > 5:
            footer_parts.append(f"...and {len(items) - 5} more items available.")
        footer_parts.append("\nWould you like to add any of these items to your cart?")
        
        if footer_parts:
            content.append({"type": "text", "text": "\n".join(footer_parts)})
        
        return content
    
    async def _build_product_content(self, item: Dict, index: int, image_formatter, 
                                   sources: List[str] = None, score: float = 0) -> List[Dict]:
        """
        Build comprehensive product content as MCP content array (text + images).
        
        Args:
            item: Product item data
            index: Product index in results
            image_formatter: ImageFormatter instance
            sources: List of sources (api, vector)
            score: Relevance score
            
        Returns:
            List of MCP content items
        """
        content = []
        
        # Handle different item formats (same logic as _format_product_basic_info)
        if "item" in item and isinstance(item["item"], dict):
            # Reranked format - extract the actual product data
            product_data = item["item"]
            if "item_details" in product_data:
                item_details = product_data["item_details"]
            else:
                item_details = product_data
        else:
            # Original format
            item_details = item.get("item_details", {})
        
        # Add product image using smart display (MCP + markdown fallback)
        image_url = self._extract_product_image(item)
        if image_url:
            name = item_details.get("descriptor", {}).get("name", "Product")
            image_content = await self._display_product_image(image_url, name, image_formatter)
            content.extend(image_content)
        
        # Build text content sections
        text_sections = []
        
        # Basic info with source badges
        basic_info = self._format_product_basic_info(item, index, sources, score)
        if basic_info:
            text_sections.append(basic_info)
        
        # Specifications - extract from tags if needed
        attributes = self._extract_attributes(item)
        specs = self._format_product_specs(attributes)
        if specs:
            text_sections.append(specs)
        
        # Availability
        availability = self._format_product_availability(item_details)
        if availability:
            text_sections.append(availability)
        
        # Seller info
        contact_info = item_details.get("@ondc/org/contact_details_consumer_care", "")
        seller_info = self._format_product_seller_info(item.get("provider_details", {}), contact_info)
        if seller_info:
            text_sections.append(seller_info)
        
        # Add text content
        if text_sections:
            content.append({
                "type": "text", 
                "text": "\n\n".join(text_sections) + "\n"
            })
        
        return content
    
    def _extract_attributes(self, item: Dict) -> Dict:
        """Extract product attributes from ONDC tags structure"""
        
        # Handle different item formats (same logic as other methods)
        if "item" in item and isinstance(item["item"], dict):
            # Reranked format - extract the actual product data
            product_data = item["item"]
            # First check if attributes are directly available in reranked product
            if "attributes" in product_data and product_data["attributes"]:
                return product_data["attributes"]
            # Get item_details from reranked structure
            if "item_details" in product_data:
                item_details = product_data["item_details"]
            else:
                item_details = product_data
        else:
            # Original format - check if attributes are directly available
            if "attributes" in item and item["attributes"]:
                return item["attributes"]
            # Get item_details from original structure
            item_details = item.get("item_details", item)
        
        # Extract from ONDC tags
        attributes = {}
        tags = item_details.get("tags", [])
        
        for tag in tags:
            if tag.get("code") == "attribute":
                attr_list = tag.get("list", [])
                for attr in attr_list:
                    code = attr.get("code", "")
                    value = attr.get("value", "")
                    if code and value:
                        attributes[code] = value
        
        return attributes
    
    def _extract_product_image(self, item: Dict) -> Optional[str]:
        """Extract the best available product image URL from ONDC item data"""
        
        # Handle different item formats (same logic as other methods)
        if "item" in item and isinstance(item["item"], dict):
            # Reranked format - extract the actual product data
            product_data = item["item"]
            if "item_details" in product_data:
                item_details = product_data["item_details"]
            else:
                item_details = product_data
        else:
            # Original format
            item_details = item.get("item_details", {})
        descriptor = item_details.get("descriptor", {})
        
        # Priority 1: Main product images
        images = descriptor.get("images", [])
        if images and len(images) > 0:
            # Handle comma-separated URLs in single string
            first_image = images[0]
            if "," in first_image:
                return first_image.split(",")[0].strip()
            return first_image
        
        # Priority 2: Symbol/thumbnail
        symbol = descriptor.get("symbol")
        if symbol:
            if "," in symbol:
                return symbol.split(",")[0].strip()
            return symbol
        
        # Priority 3: Tagged images (back_image, etc.)
        tags = item_details.get("tags", [])
        for tag in tags:
            if tag.get("code") == "image":
                tag_list = tag.get("list", [])
                for item_tag in tag_list:
                    if item_tag.get("code") == "url":
                        return item_tag.get("value")
        
        return None
    
    async def _display_product_image(self, image_url: str, product_name: str, 
                                   image_formatter) -> List[Dict]:
        """Smart image display - try MCP base64 first, fallback to validated markdown"""
        if not image_url:
            return []
        
        # First try MCP image content (base64)
        try:
            mcp_image = await image_formatter.fetch_as_base64(image_url)
            if mcp_image:
                # Image successfully created
                return [mcp_image]
        except Exception as e:
            # MCP image creation failed, will try fallback
            pass
        
        # Fallback: Enhanced markdown with image validation
        try:
            if await image_formatter.validate_url(image_url):
                return [{"type": "text", "text": f" **Product Image:** [View {product_name}]({image_url})"}]
            else:
                # Image validation failed
                pass
        except Exception as e:
            # Image validation error occurred
            pass
        
        return []
    
    def _format_product_basic_info(self, item: Dict, index: int, sources: List[str] = None, score: float = 0) -> str:
        """Format basic product information (name, price, category, description)"""
        # Handle different item formats:
        # 1. Reranked format: {'item': {...}, 'sources': [...], 'scores': ...}
        # 2. Original format: {'item_details': {...}}  
        # 3. Direct format: item IS the details
        if "item" in item and isinstance(item["item"], dict):
            # Reranked format - extract the actual product data
            product_data = item["item"]
            if "item_details" in product_data:
                item_details = product_data["item_details"]
            else:
                item_details = product_data
        elif "item_details" in item:
            # Original API format
            item_details = item["item_details"]
        else:
            # Direct format - item IS the details
            item_details = item
        descriptor = item_details.get("descriptor", {})
        price_info = item_details.get("price", {})
        
        name = descriptor.get("name", "Unknown Product")
        price = price_info.get("value", "N/A")
        currency = price_info.get("currency", "INR")
        category = item_details.get("category_id", "")
        
        # Build basic info sections
        sections = []
        
        # Product title and price with source badges
        price_formatted = f"{currency} {price:,}" if isinstance(price, (int, float)) else f"{currency} {price}"
        title = f"**{index}. {name}** - {price_formatted}"
        
        # Add source badges if available
        if sources:
            badges = []
            if "api" in sources:
                badges.append("API")
            if "vector" in sources:
                badges.append("Vector")
            if score > 0:
                badges.append(f"Score: {score:.2f}")
            if badges:
                title += f" [{' | '.join(badges)}]"
        
        sections.append(title)
        
        # Category
        if category:
            sections.append(f" **Category:** {category}")
        
        # Description
        short_desc = descriptor.get("short_desc", "").strip()
        long_desc = descriptor.get("long_desc", "").strip()
        
        if short_desc and short_desc.lower() not in ['description', '', 'n/a']:
            sections.append(f" **Description:** {short_desc}")
        elif long_desc and long_desc.lower() not in ['long', 'description', '', 'n/a']:
            sections.append(f" **Description:** {long_desc}")
        
        return "\n".join(sections)
    
    def _format_product_specs(self, attributes: Dict) -> str:
        """Format product specifications dynamically"""
        if not attributes:
            return ""
        
        sections = [" **Specifications:**"]
        
        # Priority attributes to show first
        priority_attrs = ['screen_size', 'ram', 'storage', 'cpu', 'gpu', 'brand', 'model_year']
        priority_specs = []
        other_specs = []
        
        for attr_name, attr_value in attributes.items():
            if not attr_value or str(attr_value).lower() in ['', 'null', 'none', 'n/a']:
                continue
                
            # Skip unit fields - they're handled with their base attribute
            if attr_name.endswith('_unit'):
                continue
                
            # Format attribute name nicely
            formatted_name = attr_name.replace('_', ' ').title()
            
            # Handle units properly
            if f"{attr_name}_unit" in attributes:
                unit = attributes[f"{attr_name}_unit"]
                formatted_value = f"{attr_value} {unit.upper()}" if unit else str(attr_value)
            else:
                # Special formatting for known attributes
                if attr_name == 'screen_size':
                    formatted_value = f"{attr_value} inches"
                elif attr_name in ['ram', 'rom', 'storage'] and not any(u in str(attr_value).lower() for u in ['gb', 'mb', 'tb']):
                    # Add GB if not present
                    formatted_value = f"{attr_value} GB"
                else:
                    formatted_value = str(attr_value)
            
            spec_line = f"{formatted_name}: {formatted_value}"
            
            # Categorize specs
            if attr_name in priority_attrs:
                priority_specs.append((priority_attrs.index(attr_name), spec_line))
            else:
                other_specs.append(spec_line)
        
        # Sort priority specs by their priority order
        priority_specs.sort(key=lambda x: x[0])
        priority_lines = [spec[1] for spec in priority_specs]
        
        # Format output
        if priority_lines:
            # Group priority specs in lines of 3
            for i in range(0, len(priority_lines), 3):
                group = priority_lines[i:i+3]
                sections.append(f"• {' | '.join(group)}")
        
        if other_specs:
            # Group other specs in lines of 3
            for i in range(0, len(other_specs), 3):
                group = other_specs[i:i+3]
                sections.append(f"• {' | '.join(group)}")
        
        return "\n".join(sections) if len(sections) > 1 else ""
    
    def _format_product_availability(self, item_details: Dict) -> str:
        """Format product availability and policies"""
        quantity_info = item_details.get("quantity", {})
        
        availability_parts = []
        policy_parts = []
        
        # Stock information
        if quantity_info.get("available", {}).get("count"):
            availability_parts.append(f"Stock: {quantity_info['available']['count']} available")
        if quantity_info.get("maximum", {}).get("count"):
            availability_parts.append(f"Max order: {quantity_info['maximum']['count']}")
        
        # ONDC-specific policies
        if item_details.get("@ondc/org/available_on_cod"):
            policy_parts.append("COD: ")
        if item_details.get("@ondc/org/returnable"):
            policy_parts.append("Returnable: ")
        if item_details.get("@ondc/org/return_window"):
            return_window = item_details["@ondc/org/return_window"].replace("PT", "").replace("H", " Hour")
            policy_parts.append(f"Return window: {return_window}")
        
        if not availability_parts and not policy_parts:
            return ""
        
        sections = [" **Availability:**"]
        if availability_parts:
            sections.append(f"• {' | '.join(availability_parts)}")
        if policy_parts:
            sections.append(f"• {' | '.join(policy_parts)}")
        
        return "\n".join(sections)
    
    def _format_product_seller_info(self, provider_details: Dict, contact_info: str = "") -> str:
        """Format seller information"""
        if not provider_details.get("descriptor", {}).get("name"):
            return ""
        
        provider_name = provider_details["descriptor"]["name"]
        sections = [f" **Seller:** {provider_name}"]
        
        if contact_info:
            sections.append(f"• Contact: {contact_info}")
        
        return "\n".join(sections)